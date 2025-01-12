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


host_address = 'http://localhost:6000'
host_process = None
host_address_generate = f'{host_address}/generate'
host_address_initialize = f'{host_address}/initialize'


# TODO:
# implement lora loader


class DFPipelineConfig:
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
                        "bf16",
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
                "enable_model_cpu_offload": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
                "enable_sequential_cpu_offload": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
                "enable_tiling": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
                "enable_slicing": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
                "xformers_efficient": (
                    "BOOLEAN",
                    {
                        "default": False,
                    }
                ),
            }
        }

    RETURN_TYPES = ("DF_CONFIG",)
    FUNCTION = "get_config"
    CATEGORY = "Distrifuser"        

    def get_config(
        self, model, width, height, scheduler, pipeline_type, variant,
        nproc_per_node, parallelism, no_split_batch, warmup_steps, enable_model_cpu_offload,
        enable_sequential_cpu_offload, enable_tiling, enable_slicing, xformers_efficient
    ):
        return (
            {
                "model": model,
                "width": width,
                "height": height,
                "scheduler": scheduler,
                "pipeline_type": pipeline_type,
                "variant": variant,
                "nproc_per_node": nproc_per_node,
                "parallelism": parallelism,
                "no_split_batch": no_split_batch,
                "warmup_steps": warmup_steps,
                "enable_model_cpu_offload": enable_model_cpu_offload,
                "enable_sequential_cpu_offload": enable_sequential_cpu_offload,
                "enable_tiling": enable_tiling,
                "enable_slicing": enable_slicing,
                "xformers_efficient": xformers_efficient,
            },
        )


class DFSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config": ("DF_CONFIG",),
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

    def generate(self, config, positive, negative, seed, steps, cfg, clip_skip):
        launch_host_process(config)
        data = {
            "positive_prompt": positive,  
            "negative_prompt": negative,          
            "num_inference_steps": steps,
            "seed": seed,
            "cfg": cfg,
            "clip_skip": clip_skip,
        }
        response = requests.post(host_address_generate, json=data)
        response_data = response.json()
        output_base64 = response_data.get("output")
        close_host_process()
        if output_base64:
            output_bytes = base64.b64decode(output_base64)
            output = pickle.loads(output_bytes)
            print("Media generated")
        else:
            print("No image generated")
            return (None,)
        image = output.images[0]
        tensor_image = ToTensor()(image)                    # CHW
        tensor_image = tensor_image.unsqueeze(0)            # CHW -> NCHW
        tensor_image = tensor_image.permute(0, 2, 3, 1)     # NCHW -> NHWC
        return (tensor_image,)


def launch_host_process(config):
    global host_process

    if host_process is not None:
        close_host_process()

    cmd = [
        "torchrun",
        f'--nproc_per_node={config.get("nproc_per_node")}',
        f'{cwd}/host.py',

        '--host_mode=comfyui',
        f'--model_path={checkpoints_dir}/{config.get("model")}',
        f'--width={config.get("width")}',
        f'--height={config.get("height")}',
        f'--scheduler={config.get("scheduler")}',
        f'--pipeline_type={config.get("pipeline_type")}',
        f'--variant={config.get("variant")}',
        f'--parallelism={config.get("parallelism")}',
        f'--warmup_steps={config.get("warmup_steps")}',
        '--no_cuda_graph',
        '--compel',
    ]

    if config.get("enable_model_cpu_offload"):
        cmd.append('--enable_model_cpu_offload')

    if config.get("enable_sequential_cpu_offload"):
        cmd.append('--enable_sequential_cpu_offload')

    if config.get("enable_tiling"):
        cmd.append('--enable_tiling')

    if config.get("enable_slicing"):
        cmd.append('--enable_slicing')

    if config.get("xformers_efficient"):
        cmd.append('--xformers_efficient')

    if config.get("no_split_batch"):
        cmd.append('--no_split_batch')

    host_process = subprocess.Popen(cmd)

    connection_attempts = 0
    while True:
        try:
            response = requests.get(host_address_initialize)
            if response.status_code == 200 and response.json().get("status") == "initialized":
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
        connection_attempts += 1
        if connection_attempts > 60:
            assert False, "Failed to launch host. Check logs for details."


def close_host_process():
    global host_process
    if host_process is not None:
        host_process.terminate()
        host_process = None
    return


NODE_CLASS_MAPPINGS = {
    "DFPipelineConfig": DFPipelineConfig,
    "DFSampler": DFSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "DFPipelineConfig": "DFPipelineConfig",
    "DFSampler": "DFSampler",
}
