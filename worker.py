# https://github.com/mit-han-lab/distrifuser/blob/main/scripts/run_sdxl.py

import argparse
import os
import time
import torch


from diffusers.schedulers import (
    # DDIM
    DDIMScheduler,
    # DPM
    DPMSolverMultistepScheduler,
    # Euler
    EulerDiscreteScheduler,
)


from tqdm import trange
from distrifuser.pipelines import DistriSDPipeline, DistriSDXLPipeline
from distrifuser.utils import DistriConfig


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="generation",
        choices=["generation", "benchmark"],
        help="Purpose of running the script",
    )

    # Diffuser specific arguments
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--scheduler", type=str, default="ddim", choices=["euler", "dpm-solver", "ddim"])
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # DistriFuser specific arguments
    parser.add_argument(
        "--no_split_batch", action="store_true", help="Disable the batch splitting for classifier-free guidance"
    )
    parser.add_argument("--warmup_steps", type=int, default=4, help="Number of warmup steps")
    parser.add_argument(
        "--sync_mode",
        type=str,
        default="corrected_async_gn",
        choices=["separate_gn", "stale_gn", "corrected_async_gn", "sync_gn", "full_sync", "no_sync"],
        help="Different GroupNorm synchronization modes",
    )
    parser.add_argument(
        "--parallelism",
        type=str,
        default="patch",
        choices=["patch", "tensor", "naive_patch"],
        help="patch parallelism, tensor parallelism or naive patch",
    )
    parser.add_argument("--no_cuda_graph", action="store_true", help="Disable CUDA graph")
    parser.add_argument(
        "--split_scheme",
        type=str,
        default="alternate",
        choices=["row", "col", "alternate"],
        help="Split scheme for naive patch",
    )

    # Benchmark specific arguments
    parser.add_argument("--output_type", type=str, default="pil", choices=["latent", "pil"])
    parser.add_argument("--warmup_times", type=int, default=5, help="Number of warmup times")
    parser.add_argument("--test_times", type=int, default=20, help="Number of test times")
    parser.add_argument(
        "--ignore_ratio", type=float, default=0.2, help="Ignored ratio of the slowest and fastest steps"
    )

    # Added arguments
    parser.add_argument("--positive_prompt", type=str, default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None, help="Path to model folder")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--variant", type=str, default="fp16", help="PyTorch variant [fp16/fp32]")
    parser.add_argument("--pipeline_type", type=str, default="SDXL", help="Stable Diffusion pipeline type [SD/SDXL]")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    assert args.model_path is not None
    distri_config = DistriConfig(
        height=args.height,
        width=args.width,
        do_classifier_free_guidance=args.guidance_scale > 1,
        split_batch=not args.no_split_batch,
        warmup_steps=args.warmup_steps,
        mode=args.sync_mode,
        use_cuda_graph=not args.no_cuda_graph,
        parallelism=args.parallelism,
        split_scheme=args.split_scheme,
    )

    pretrained_model_name_or_path = args.model_path
    if args.scheduler == "euler":
        scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    elif args.scheduler == "dpm-solver":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    elif args.scheduler == "ddim":
        scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    else:
        raise NotImplementedError

    assert args.variant in ["fp16", "fp32"], "Unsupported variant"
    if args.variant == "fp16":
        torchType = torch.float16
        variant = "fp16"
    else:
        torchType = torch.float32
        variant = None

    assert args.pipeline_type in ["SD", "SDXL"], "Unsupported pipeline"
    if args.pipeline_type == "SDXL":
        pipe = DistriSDXLPipeline
    else:
        pipe = DistriSDPipeline

    pipeline = pipe.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        distri_config=distri_config,
        torch_dtype=torchType,
        variant=variant,
        use_safetensors=True,
        scheduler=scheduler,
    )

    if args.mode == "generation":
        assert args.output_path is not None
        pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
        image = pipeline(
            prompt=args.positive_prompt,
            negative_prompt=args.negative_prompt,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).images[0]
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        image.save(args.output_path)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
