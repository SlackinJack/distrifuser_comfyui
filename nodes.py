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


host_address = 'http://localhost:6000'
host_process = None
host_address_generate = f'{host_address}/generate'
host_address_initialize = f'{host_address}/initialize'


cwd = os.path.dirname(__file__)
comfy_root = os.path.dirname(os.path.dirname(cwd))
checkpoints_dir = os.path.join(os.path.join(comfy_root, "models"), "checkpoints")
outputs_dir = os.path.join(comfy_root, "output")
models = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]


INT_MAX = 2 ** 32 - 1
INT_MIN = -1 * INT_MAX
CONFIG =                ("DF_CONFIG",)
BOOLEAN_DEFAULT_FALSE = ("BOOLEAN", { "default": False })
MODEL_LIST =            (models,)


SCHEDULERS = (
    list(["ddim", "euler", "euler_a", "dpm_2", "dpm_2_a", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "heun", "lms", "pndm", "unipc"]),
    { "default": "dpmpp_2m" }
)
PIPELINE_TYPE =         (["SD", "SDXL"],                        { "default": "SDXL" })
VARIANT =               (["bf16", "fp16", "fp32"],              { "default": "fp16" })
PARALLELISM =           (["naive_patch", "patch", "tensor"],    { "default": "patch" })
NPROC_PER_NODE =        ("INT",                                 { "default": 2,     "min": 1,       "max": INT_MAX, "step": 1 })


PROMPT =                ("STRING",                              { "default": "",    "multiline": True })
RESOLUTION =            ("INT",                                 { "default": 512,   "min": 8,       "max": INT_MAX, "step": 8 })
SEED =                  ("INT",                                 { "default": 0,     "min": 0,       "max": INT_MAX, "step": 1 })
STEPS =                 ("INT",                                 { "default": 60,    "min": 1,       "max": INT_MAX, "step": 1 })
CFG =                   ("FLOAT",                               { "default": 3.5,   "min": 0,       "max": INT_MAX, "step": 0.1 })
CLIP_SKIP =             ("INT",                                 { "default": 1,     "min": 0,       "max": INT_MAX, "step": 1 })


# TODO:
# implement lora loader


class DFPipelineConfig:
    @classmethod
    def INPUT_TYPES(s):
        models = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
        return {
            "required": {
                "model":                            MODEL_LIST,
                "width":                            RESOLUTION,
                "height":                           RESOLUTION,
                "scheduler":                        SCHEDULERS,
                "pipeline_type":                    PIPELINE_TYPE,
                "variant":                          VARIANT,
                "nproc_per_node":                   NPROC_PER_NODE,
                "parallelism":                      PARALLELISM,
                "no_split_batch":                   BOOLEAN_DEFAULT_FALSE,
                "enable_model_cpu_offload":         BOOLEAN_DEFAULT_FALSE,
                "enable_sequential_cpu_offload":    BOOLEAN_DEFAULT_FALSE,
                "enable_tiling":                    BOOLEAN_DEFAULT_FALSE,
                "enable_slicing":                   BOOLEAN_DEFAULT_FALSE,
                "xformers_efficient":               BOOLEAN_DEFAULT_FALSE,
            }
        }

    RETURN_TYPES = ("DF_CONFIG",)
    FUNCTION = "get_config"
    CATEGORY = "Distrifuser"        

    def get_config(
        self, model, width, height, scheduler, pipeline_type, variant,
        nproc_per_node, parallelism, no_split_batch, enable_model_cpu_offload,
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
                "warmup_steps": 10,
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
                "config":       CONFIG,
                "positive":     PROMPT,
                "negative":     PROMPT,
                "seed":         SEED,
                "steps":        STEPS,
                "cfg":          CFG,
                "clip_skip":    CLIP_SKIP,
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
