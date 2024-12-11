# https://github.com/xdit-project/xDiT/blob/1c31746e2f903e791bc2a41a0bc23614958e46cd/comfyui-xdit/nodes.py

import numpy
import os
import random
import subprocess
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor


from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig


cwd = os.path.dirname(__file__)
comfy_root = os.path.dirname(os.path.dirname(cwd))
checkpoints_dir = os.path.join(os.path.join(comfy_root, "models"), "checkpoints")
outputs_dir = os.path.join(comfy_root, "output")


class DistrifuserSampler:
    @classmethod
    def INPUT_TYPES(s):
        models = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
        return {
            "required": {
                "model": (models,), 
                "positive": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": ""
                    }
                ),
                "negative": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": ""
                    }
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2 ** 32 - 1
                    }
                ),
                "steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 1024
                    }
                ),
                "scheduler": (
                    list([
                        "ddim",
                        "dpm-solver",
                        "euler",
                    ]),
                    {
                        "default": "euler",
                    }
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.1
                    }
                ),
                "variant": (
                    list([
                        "fp16",
                        "fp32"
                    ]),
                    {
                        "default": "fp16",
                    }
                ),
                "pipeline_type": (
                    list([
                        "SD",
                        "SDXL"
                    ]),
                    {
                        "default": "SDXL",
                    }
                ),
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 8192,
                        "step": 8
                    }
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 8192,
                        "step": 8
                    }
                ),
                "warmup_steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 32,
                        "step": 1
                    }
                ),
                "parallelism": (
                    list([
                        "naive_patch",
                        "patch",
                        "tensor",
                    ]),
                    {
                        "default": "patch",
                    }
                ),
                "nproc_per_node": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8192,
                        "step": 1
                    }
                ),
            }
        }


    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Distrifuser"


    def generate(self, model, positive, negative, seed, steps, scheduler, cfg, variant, pipeline_type, width, height, warmup_steps, parallelism, nproc_per_node):
        file_name = str(random.randint(1, 2 ** 32 - 1)) + ".png"
        output_path = f"{outputs_dir}/{file_name}"
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            f"{cwd}/worker.py",
            f"--model_path={checkpoints_dir}/{model}",
            f"--output_path={output_path}",
            f"--positive_prompt={positive}",
            f"--negative_prompt={negative}",
            f"--num_inference_steps={steps}",
            f"--height={height}",
            f"--width={width}",
            f"--scheduler={scheduler}",
            f"--seed={seed}",
            f"--guidance_scale={cfg}",
            f"--pipeline_type={pipeline_type}",
            f"--variant={variant}",
            f"--warmup_steps={warmup_steps}",
            f"--parallelism={parallelism}",
        ]

        subprocess.run(cmd, check=True)

        if os.path.exists(output_path):
            print("\nImage generated: " + str(output_path))
            image = Image.open(output_path)
            tensor_image = ToTensor()(image)
            tensor_image = tensor_image.unsqueeze(0)
            tensor_image = tensor_image.permute(0, 2, 3, 1).cpu().float()
            return (tensor_image,)
        else:
            print("\nImage generation failed.")
            return None


NODE_CLASS_MAPPINGS = {
    "DistrifuserSampler": DistrifuserSampler
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "DistrifuserSampler": "DistrifuserSampler"
}
