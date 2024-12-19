# https://github.com/xdit-project/xDiT/blob/1c31746e2f903e791bc2a41a0bc23614958e46cd/comfyui-xdit/nodes.py

import base64
import os
import pickle
import requests
import subprocess
import time
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor


cwd = os.path.dirname(__file__)
comfy_root = os.path.dirname(os.path.dirname(cwd))
checkpoints_dir = os.path.join(os.path.join(comfy_root, "models"), "checkpoints")
outputs_dir = os.path.join(comfy_root, "output")


# TODO:
# implement lora loader


class DistrifuserPipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        models = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
        return {
            "required": {
                "model": (models,), 
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
                "scheduler": (
                    list([
                        "ddim",
                        "euler",
                        "euler_a",
                        "dpm_2",
                        "dpm_2_a",
                        "dpmpp_2m",
                        "dpmpp_2m_sde",
                        "dpmpp_sde",
                        "heun",
                        "lms",
                        "pndm",
                        "unipc",
                    ]),
                    {
                        "default": "dpmpp_2m",
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
                "pipeline_type": (
                    list([
                        "SD",
                        "SDXL"
                    ]),
                    {
                        "default": "SDXL",
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
                "nproc_per_node": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8192,
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
                "low_vram": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
                "no_split_batch": (
                    "BOOLEAN",
                    {
                        "default": False,
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
            }
        }

    RETURN_TYPES = ("DISTRIFUSER_PIPELINE",)
    FUNCTION = "launch_host"
    CATEGORY = "Distrifuser"        

    def launch_host(self, model, width, height, scheduler, cfg, pipeline_type, variant, nproc_per_node, parallelism, low_vram, no_split_batch, warmup_steps):
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            f"{cwd}/host.py",

            f"--model_path={checkpoints_dir}/{model}",
            f"--width={width}",
            f"--height={height}",
            f"--scheduler={scheduler}",
            f"--guidance_scale={cfg}",
            f"--pipeline_type={pipeline_type}",
            f"--variant={variant}",
            f"--parallelism={parallelism}",
            f"--warmup_steps={warmup_steps}",
            '--no_cuda_graph',
            '--compel',
        ]

        if low_vram:
            # cmd.append('--enable_model_cpu_offload')          # breaks parallelism
            # cmd.append('--enable_sequential_cpu_offload')     # crash
            cmd.append('--enable_tiling')
            cmd.append('--enable_slicing')

        # enable for more vram usage, and slower
        # best to leave this disabled
        if no_split_batch:
            cmd.append('--no_split_batch')

        process = subprocess.Popen(cmd)
        host = 'http://localhost:6000'
        while True:
            try:
                response = requests.get(f'{host}/initialize')
                if response.status_code == 200 and response.json().get("status") == "initialized":
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        return (f"{host}/generate", )


class DistrifuserSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("DISTRIFUSER_PIPELINE",),
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
                "clip_skip": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 1024
                    }
                ),
            }
        }


    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Distrifuser"

    def generate(self, pipeline, positive, negative, seed, steps, clip_skip):
        url = pipeline
        data = {
            "positive_prompt": positive,  
            "negative_prompt": negative,          
            "num_inference_steps": steps,
            "seed": seed,
            #"cfg": cfg,
            "clip_skip": clip_skip,
        }
        response = requests.post(url, json=data)
        response_data = response.json()
        output_base64 = response_data.get("output", "")
        if output_base64:
            output_bytes = base64.b64decode(output_base64)
            output = pickle.loads(output_bytes)
            print("Media generated")
        else:
            print("No image generated")
            return (None,)
        image = output.images[0]
        tensor_image = ToTensor()(image)
        tensor_image = tensor_image.unsqueeze(0)
        tensor_image = tensor_image.permute(0, 2, 3, 1).cpu().float()
        return (tensor_image,)


NODE_CLASS_MAPPINGS = {
    "DistrifuserPipelineLoader": DistrifuserPipelineLoader,
    "DistrifuserSampler": DistrifuserSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "DistrifuserPipelineLoader": "DistrifuserPipelineLoader",
    "DistrifuserSampler": "DistrifuserSampler",
}
