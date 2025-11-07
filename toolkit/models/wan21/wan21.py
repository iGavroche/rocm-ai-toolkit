# WIP, coming soon ish
from functools import partial
import torch
import yaml
from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.memory_management.manager import MemoryManager
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import  WanPipeline, WanTransformer3DModel, AutoencoderKL
from .autoencoder_kl_wan import AutoencoderKLWan
import os
import sys

import weakref
import torch
import yaml

from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds

import os
import copy
from toolkit.config_modules import ModelConfig, GenerateImageConfig, ModelArch
import torch
from optimum.quanto import freeze, qfloat8, QTensor, qint4
from toolkit.util.quantize import quantize, get_qtype
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from typing import TYPE_CHECKING, List
from toolkit.accelerator import unwrap_model
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.pipelines.wan.pipeline_wan import XLA_AVAILABLE
# from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from typing import Any, Callable, Dict, List, Optional, Union
from toolkit.models.wan21.wan_lora_convert import convert_to_diffusers, convert_to_original
from toolkit.util.quantize import quantize_model
from toolkit.models.loaders.umt5 import get_umt5_encoder

# for generation only?
scheduler_configUniPC = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.33.0.dev0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False
}

# for training. I think it is right
scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": False
}


class AggressiveWanUnloadPipeline(WanPipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer_2: Optional[WanTransformer3DModel] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,  # Wan2.2 ti2v
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            transformer_2=transformer_2,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
            vae=vae,
            scheduler=scheduler,
        )
        self._exec_device = device
    @property
    def _execution_device(self):
        return self._exec_device
    
    def __call__(
        self: WanPipeline,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None],
                  PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # unload vae and transformer
        vae_device = self.vae.device
        transformer_device = self.transformer.device
        text_encoder_device = self.text_encoder.device
        device = self.transformer.device
        
        print("Unloading vae")
        self.vae.to("cpu")
        self.text_encoder.to(device)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # unload text encoder
        print("Unloading text encoder")
        self.text_encoder.to("cpu")

        self.transformer.to(device)

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(device, transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device, transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - \
            num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(device, transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * \
                        (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop(
                        "prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # unload transformer
        # load vae
        print("Loading Vae")
        self.vae.to(vae_device)

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(
                video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)


class Wan21(BaseModel):
    arch = 'wan21'
    _wan_generation_scheduler_config = scheduler_configUniPC
    _wan_expand_timesteps = False
    _wan_vae_path = None
    
    _comfy_te_file = ['text_encoders/umt5_xxl_fp16.safetensors', 'text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors']
    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='bf16',
            custom_pipeline=None,
            noise_scheduler=None,
            **kwargs
    ):
        super().__init__(device, model_config, dtype,
                         custom_pipeline, noise_scheduler, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ['WanTransformer3DModel']

        # cache for holding noise
        self.effective_noise = None
        
    def get_bucket_divisibility(self):
        return 16

    # static method to get the scheduler
    @staticmethod
    def get_train_scheduler():
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
        return scheduler
    
    def load_wan_transformer(self, transformer_path, subfolder=None):
        self.print_and_status_update(f"Loading transformer from: {transformer_path}")
        if subfolder:
            self.print_and_status_update(f"  Subfolder: {subfolder}")
        self.print_and_status_update("  This may download from HuggingFace if not cached...")
        dtype = self.torch_dtype
        try:
            transformer = WanTransformer3DModel.from_pretrained(
                transformer_path,
                subfolder=subfolder,
                torch_dtype=dtype,
            ).to(dtype=dtype)
            self.print_and_status_update("  Transformer loaded successfully")
        except Exception as e:
            self.print_and_status_update(f"  ERROR loading transformer: {e}")
            self.print_and_status_update(f"  This could be due to:")
            self.print_and_status_update(f"    - Network issues preventing download from HuggingFace")
            self.print_and_status_update(f"    - Insufficient disk space")
            self.print_and_status_update(f"    - Invalid model path: {transformer_path}")
            raise

        if self.model_config.split_model_over_gpus:
            raise ValueError(
                "Splitting model over gpus is not supported for Wan2.1 models")

        if self.model_config.low_vram:
            # quantize on the device
            transformer.to('cpu', dtype=dtype)
            flush()
        else:
            transformer.to(self.device_torch, dtype=dtype)
            flush()

        if self.model_config.assistant_lora_path is not None or self.model_config.inference_lora_path is not None:
            raise ValueError(
                "Assistant LoRA is not supported for Wan2.1 models currently")

        if self.model_config.lora_path is not None:
            raise ValueError(
                "Loading LoRA is not supported for Wan2.1 models currently")

        flush()
        
        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()
        
        if self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0:
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent
            )
        
        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to('cpu')

        return transformer

    def load_model(self):
        dtype = self.torch_dtype
        model_path = self.model_config.name_or_path

        self.print_and_status_update(f"Loading Wan model from: {model_path}")
        self.print_and_status_update("Note: This may download a large model from HuggingFace if not already cached. This can take several minutes.")
        subfolder = 'transformer'
        transformer_path = model_path
        if os.path.exists(transformer_path):
            subfolder = None
            transformer_path = os.path.join(transformer_path, 'transformer')
        
        te_path = "ai-toolkit/umt5_xxl_encoder"   
        if os.path.exists(os.path.join(model_path, 'text_encoder')):
            te_path = model_path
        
        vae_path = self.model_config.extras_name_or_path
        if os.path.exists(os.path.join(model_path, 'vae')):
            vae_path = model_path

        transformer = self.load_wan_transformer(
            transformer_path,
            subfolder=subfolder,
        )

        flush()

        self.print_and_status_update(f"Loading UMT5EncoderModel from: {te_path}")
        self.print_and_status_update("  This may download from HuggingFace if not cached...")
        
        try:
            tokenizer, text_encoder = get_umt5_encoder(
                model_path=te_path,
                tokenizer_subfolder="tokenizer",
                encoder_subfolder="text_encoder",
                torch_dtype=dtype,
                comfy_files=self._comfy_te_file
            )
            self.print_and_status_update("  UMT5EncoderModel loaded successfully")
        except Exception as e:
            self.print_and_status_update(f"  ERROR loading UMT5EncoderModel: {e}")
            raise

        # Check available GPU memory and decide dynamically
        # Text encoder is ~11 GB unquantized, ~3-4 GB quantized
        try:
            import torch
            device = torch.cuda.current_device() if torch.cuda.is_available() else None
            if device is not None:
                total_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
                reserved_mem = torch.cuda.memory_reserved(device) / 1024**3
                allocated_mem = torch.cuda.memory_allocated(device) / 1024**3
                available_mem = (total_mem - reserved_mem) * 0.9  # 90% to account for fragmentation
                self.print_and_status_update(f"GPU memory before text encoder: {available_mem:.2f} GB available / {total_mem:.2f} GB total")
                
                # Estimate text encoder size
                estimated_te_gb = 4.0 if self.model_config.quantize_te else 11.0
                
                # Clear GPU cache before attempting to move
                torch.cuda.empty_cache()
                flush()
                
                # If we're going to quantize, keep on CPU for quantization, then move to GPU
                # Otherwise try to move to GPU immediately if we have space
                if not self.model_config.quantize_te:
                    if available_mem > estimated_te_gb:
                        try:
                            self.print_and_status_update(f"Moving UMT5EncoderModel to GPU ({available_mem:.2f} GB available)")
                            text_encoder.to(self.device_torch, dtype=dtype)
                            torch.cuda.empty_cache()
                            flush()
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower() or "hip" in str(e).lower():
                                self.print_and_status_update("OOM moving text encoder to GPU - keeping on CPU")
                            else:
                                raise
                    else:
                        self.print_and_status_update(f"Insufficient GPU memory ({available_mem:.2f} GB available), keeping text encoder on CPU")

                if self.model_config.quantize_te:
                    self.print_and_status_update("Quantizing UMT5EncoderModel (on CPU)")
                    quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
                    freeze(text_encoder)
                    torch.cuda.empty_cache()
                    flush()
                    # After quantization, check memory again and move to GPU if possible
                    if device is not None:
                        available_mem = (torch.cuda.get_device_properties(device).total_memory / 1024**3 - 
                                       torch.cuda.memory_reserved(device) / 1024**3) * 0.9
                        if available_mem > estimated_te_gb:
                            try:
                                self.print_and_status_update(f"Moving quantized UMT5EncoderModel to GPU ({available_mem:.2f} GB available)")
                                text_encoder.to(self.device_torch, dtype=dtype)
                                torch.cuda.empty_cache()
                                flush()
                                self.print_and_status_update("Quantized UMT5EncoderModel moved to GPU successfully")
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower() or "hip" in str(e).lower():
                                    self.print_and_status_update("OOM moving quantized text encoder to GPU - keeping on CPU")
                                else:
                                    raise
                        else:
                            self.print_and_status_update(f"Insufficient GPU memory ({available_mem:.2f} GB available), keeping quantized text encoder on CPU")
        except Exception as e:
            # Fallback to original conservative logic
            self.print_and_status_update(f"Could not check GPU memory: {e}, using conservative approach")
            torch.cuda.empty_cache()
            flush()
            
            if not self.model_config.quantize_te:
                try:
                    self.print_and_status_update("Moving UMT5EncoderModel to GPU")
                    text_encoder.to(self.device_torch, dtype=dtype)
                    torch.cuda.empty_cache()
                    flush()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "hip" in str(e).lower():
                        self.print_and_status_update("OOM moving text encoder to GPU - keeping on CPU")
                    else:
                        raise

            if self.model_config.quantize_te:
                self.print_and_status_update("Quantizing UMT5EncoderModel (on CPU)")
                quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
                freeze(text_encoder)
                torch.cuda.empty_cache()
                flush()
                try:
                    self.print_and_status_update("Moving quantized UMT5EncoderModel to GPU")
                    text_encoder.to(self.device_torch, dtype=dtype)
                    torch.cuda.empty_cache()
                    flush()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "hip" in str(e).lower():
                        self.print_and_status_update("OOM moving quantized text encoder to GPU - keeping on CPU")
                    else:
                        raise
        
        if self.model_config.layer_offloading and self.model_config.layer_offloading_text_encoder_percent > 0:
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent
            )

        if self.model_config.low_vram:
            print("Moving transformer back to GPU")
            # we can move it back to the gpu now
            transformer.to(self.device_torch)

        scheduler = Wan21.get_train_scheduler()
        self.print_and_status_update(f"Loading VAE from: {vae_path if self._wan_vae_path is None else self._wan_vae_path}")
        self.print_and_status_update("  This may download from HuggingFace if not cached...")
        # todo, example does float 32? check if quality suffers
        
        try:
            if self._wan_vae_path is not None:
                # load the vae from individual repo
                vae = AutoencoderKLWan.from_pretrained(
                    self._wan_vae_path, torch_dtype=dtype).to(dtype=dtype)
            else:
                vae = AutoencoderKLWan.from_pretrained(
                    vae_path, subfolder="vae", torch_dtype=dtype).to(dtype=dtype)
            self.print_and_status_update("  VAE loaded successfully")
        except Exception as e:
            self.print_and_status_update(f"  ERROR loading VAE: {e}")
            raise
        flush()

        self.print_and_status_update("Making pipe")
        pipe: WanPipeline = WanPipeline(
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
        )
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer

        pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        text_encoder.to(self.device_torch)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()
        self.pipeline = pipe
        self.model = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def get_generation_pipeline(self):
        scheduler = UniPCMultistepScheduler(**self._wan_generation_scheduler_config)
        if self.model_config.low_vram:
            pipeline = AggressiveWanUnloadPipeline(
                vae=self.vae,
                transformer=self.model,
                transformer_2=self.model,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                scheduler=scheduler,
                expand_timesteps=self._wan_expand_timesteps,
                device=self.device_torch
            )
        else:
            pipeline = WanPipeline(
                vae=self.vae,
                transformer=self.unet,
                transformer_2=self.unet,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                expand_timesteps=self._wan_expand_timesteps,
                scheduler=scheduler,
            )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: WanPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)
        pipeline = pipeline.to(self.device_torch)
        # todo, figure out how to do video
        output = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="pil",
            **extra
        )[0]

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item  # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs
    ):
        # vae_scale_factor_spatial = 8
        # vae_scale_factor_temporal = 4
        # num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        # shape = (
        #     batch_size,
        #     num_channels_latents, # 16
        #     num_latent_frames,  # 81
        #     int(height) // self.vae_scale_factor_spatial,
        #     int(width) // self.vae_scale_factor_spatial,
        # )

        noise_pred = self.model(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        # Check actual device from parameters (more reliable than .device attribute)
        text_encoder_params = list(self.pipeline.text_encoder.parameters())
        if len(text_encoder_params) > 0:
            actual_device = next(iter(text_encoder_params)).device
        else:
            actual_device = torch.device('cpu')
        
        # Normalize devices for comparison (cuda:0 == cuda)
        # Convert both to string and normalize (cuda:0 -> cuda, cuda -> cuda)
        actual_str = str(actual_device)
        target_str = str(self.device_torch)
        
        # Normalize: remove :0 suffix if present, both should become 'cuda'
        if actual_str.startswith('cuda'):
            actual_normalized = 'cuda' + (actual_str.split(':')[1] if ':' in actual_str else '')
        else:
            actual_normalized = actual_str
            
        if target_str.startswith('cuda'):
            target_normalized = 'cuda' + (target_str.split(':')[1] if ':' in target_str else '')
        else:
            target_normalized = target_str
        
        # For CUDA devices, just check if they're both CUDA (index doesn't matter for single GPU)
        if actual_str.startswith('cuda') and target_str.startswith('cuda'):
            devices_match = True  # Both are CUDA, that's good enough
        else:
            devices_match = actual_device == self.device_torch
        
        if not devices_match:
            # CRITICAL: On ROCm with PYTORCH_NO_HIP_MEMORY_CACHING=1, model.to(device) can hang indefinitely
            # This is a known issue with large model moves when the caching allocator is disabled
            print(f"WARNING: Text encoder is on {actual_device}, needs to move to {self.device_torch}")
            print(f"WARNING: This move may hang on ROCm with PYTORCH_NO_HIP_MEMORY_CACHING=1")
            print(f"Attempting move - if this hangs, consider removing PYTORCH_NO_HIP_MEMORY_CACHING=1")
            
            # Clear cache before move
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Attempt the move - this may hang
            self.pipeline.text_encoder.to(self.device_torch)
            torch.cuda.synchronize()
            print(f"Text encoder moved successfully to {self.device_torch}")
        
        print(f"Calling pipeline.encode_prompt() with prompt: {prompt[:50]}...")
        print(f"  Device: {self.device_torch}, dtype: {self.torch_dtype}")
        print(f"  Text encoder device: {next(iter(self.pipeline.text_encoder.parameters())).device}")
        
        # Check memory before encoding
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
            reserved_mem = torch.cuda.memory_reserved(device) / 1024**3
            allocated_mem = torch.cuda.memory_allocated(device) / 1024**3
            free_mem = total_mem - reserved_mem
            print(f"  Memory before encode_prompt: {free_mem:.2f} GB free / {total_mem:.2f} GB total")
        
        print(f"  Starting encode_prompt call (this may take time on ROCm)...")
        
        # WORKAROUND: On ROCm with PYTORCH_NO_HIP_MEMORY_CACHING=1, pipeline.encode_prompt() can hang
        # Bypass the pipeline and call the text encoder directly (same as isolated test)
        use_direct_call = os.environ.get('PYTORCH_NO_HIP_MEMORY_CACHING') == '1'
        
        if use_direct_call:
            print(f"  WORKAROUND: Bypassing pipeline.encode_prompt() and calling text encoder directly")
            print(f"  This avoids a known ROCm hang in pipeline.encode_prompt()")
            
            try:
                # Get tokenizer and text encoder (this should be fast)
                print(f"  Getting tokenizer and text encoder from pipeline...")
                tokenizer = self.pipeline.tokenizer
                print(f"  Tokenizer retrieved: {type(tokenizer).__name__}")
                text_encoder = self.pipeline.text_encoder
                print(f"  Text encoder retrieved: {type(text_encoder).__name__}")
                
                # Tokenize
                print(f"  Tokenizing prompt...")
                
                # Check text encoder state
                print(f"  Text encoder type: {type(text_encoder).__name__}")
                print(f"  Text encoder device: {next(iter(text_encoder.parameters())).device}")
                is_quantized = False
                try:
                    from toolkit.util.quantize import is_model_quantized
                    is_quantized = is_model_quantized(text_encoder)
                    print(f"  Text encoder is quantized: {is_quantized}")
                except Exception as e:
                    print(f"  Could not check quantization status: {e}")
                    is_quantized = False
                
                # Convert prompt to list if needed
                if isinstance(prompt, str):
                    prompts = [prompt]
                else:
                    prompts = prompt
                
                # Call encode_prompts_auraflow directly (same as pipeline.encode_prompt does internally)
                print(f"  Calling encode_prompts_auraflow directly...")
                print(f"  Step 1: About to tokenize...")
                
                # Tokenize first (this should be fast)
                text_inputs = tokenizer(
                    prompts,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt",
                )
                print(f"  Step 2: Tokenization complete, moving to device...")
                
                # Move to device
                device = text_encoder.device if hasattr(text_encoder, 'device') else next(iter(text_encoder.parameters())).device
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                print(f"  Step 3: Inputs moved to {device}, about to call text_encoder.forward()...")
                print(f"  Step 4: This is where it may hang on ROCm...")
                
                # Call text encoder directly with explicit diagnostics
                import time
                start_time = time.time()
                print(f"  Step 5: Starting text_encoder forward pass at {start_time:.2f}...")
                
                # WORKAROUND: If quantized, run on CPU to avoid ROCm hang
                # We'll use a temporary copy to avoid modifying the pipeline's text encoder
                text_encoder_original_device = next(iter(text_encoder.parameters())).device
                run_on_cpu = is_quantized and text_encoder_original_device.type == 'cuda'
                
                if run_on_cpu:
                    print(f"  WORKAROUND: Text encoder is quantized - running forward pass on CPU")
                    print(f"  This avoids a ROCm hang with quantized models + PYTORCH_NO_HIP_MEMORY_CACHING=1")
                    # Move inputs to CPU (don't modify the model itself)
                    text_inputs_cpu = {k: v.cpu() for k, v in text_inputs.items()}
                    print(f"  Moved inputs to CPU for forward pass")
                else:
                    text_inputs_cpu = text_inputs
                
                try:
                    with torch.no_grad():
                        # Call forward pass directly
                        print(f"  Step 5a: Calling text_encoder.forward()...")
                        if run_on_cpu:
                            # Temporarily move model to CPU for forward pass
                            self.pipeline.text_encoder = self.pipeline.text_encoder.cpu()
                        encoder_outputs = self.pipeline.text_encoder(**text_inputs_cpu)
                        
                        # Extract hidden states properly (UMT5 returns BaseModelOutputWithPastAndCrossAttentions)
                        if hasattr(encoder_outputs, 'last_hidden_state'):
                            prompt_embeds = encoder_outputs.last_hidden_state
                        elif isinstance(encoder_outputs, tuple):
                            prompt_embeds = encoder_outputs[0]
                        else:
                            prompt_embeds = encoder_outputs
                        
                        print(f"  Step 5b: Forward pass returned, shape: {prompt_embeds.shape}")
                        
                        # Move result back to GPU if we ran on CPU
                        if run_on_cpu:
                            print(f"  Moving prompt_embeds back to GPU...")
                            prompt_embeds = prompt_embeds.to(device=text_encoder_original_device)
                            print(f"  Prompt embeds moved to {text_encoder_original_device}")
                finally:
                    # ALWAYS restore model to original device if we moved it
                    if run_on_cpu:
                        current_device = next(iter(self.pipeline.text_encoder.parameters())).device
                        if current_device != text_encoder_original_device:
                            self.pipeline.text_encoder = self.pipeline.text_encoder.to(text_encoder_original_device)
                            print(f"  Restored text encoder to {text_encoder_original_device}")
                
                elapsed = time.time() - start_time
                print(f"  Step 6: Text encoder forward pass completed in {elapsed:.2f} seconds")
                
                # Create attention mask
                prompt_attention_mask = text_inputs["attention_mask"].unsqueeze(-1).expand(prompt_embeds.shape)
                prompt_embeds = prompt_embeds * prompt_attention_mask
                print(f"  Step 7: Attention mask applied")
                
                # Move to correct device/dtype
                prompt_embeds = prompt_embeds.to(device=self.device_torch, dtype=self.torch_dtype)
                print(f"  Direct encode completed successfully")
                print(f"  Prompt embeds shape: {prompt_embeds.shape}")
                
            except Exception as e:
                print(f"  ERROR in direct encode: {e}")
                print(f"  Falling back to pipeline.encode_prompt()...")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                # Fall through to pipeline.encode_prompt
                use_direct_call = False
        
        if not use_direct_call:
            try:
                prompt_embeds, _ = self.pipeline.encode_prompt(
                    prompt,
                    do_classifier_free_guidance=False,
                    max_sequence_length=512,
                    device=self.device_torch,
                    dtype=self.torch_dtype,
                )
                print(f"  encode_prompt completed successfully")
                print(f"  Prompt embeds shape: {prompt_embeds.shape if hasattr(prompt_embeds, 'shape') else 'N/A'}")
            except Exception as e:
                print(f"  ERROR in encode_prompt: {e}")
                print(f"  Error type: {type(e).__name__}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                raise
        
        # Extract prompt_embeds from tuple if needed
        if isinstance(prompt_embeds, tuple):
            prompt_embeds = prompt_embeds[0]
        
        return PromptEmbeds(prompt_embeds)

    @torch.no_grad()
    def encode_images(
            self,
            image_list: List[torch.Tensor],
            device=None,
            dtype=None
    ):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device('cpu'):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        image_list = [image.to(device, dtype=dtype) for image in image_list]

        # Normalize shapes
        norm_images = []
        for image in image_list:
            if image.ndim == 3:
                # (C, H, W) -> (C, 1, H, W)
                norm_images.append(image.unsqueeze(1))
            elif image.ndim == 4:
                # (T, C, H, W) -> (C, T, H, W)
                norm_images.append(image.permute(1, 0, 2, 3))
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")

        # Stack to (B, C, T, H, W)
        images = torch.stack(norm_images)
        B, C, T, H, W = images.shape

        # Resize if needed (B * T, C, H, W)
        if H % 8 != 0 or W % 8 != 0:
            target_h = H // 8 * 8
            target_w = W // 8 * 8
            images = images.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            images = F.interpolate(images, size=(target_h, target_w), mode='bilinear', align_corners=False)
            images = images.view(B, T, C, target_h, target_w).permute(0, 2, 1, 3, 4)

        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * latents_std

        return latents.to(device, dtype=dtype)

    def get_model_has_grad(self):
        return self.model.proj_out.weight.requires_grad

    def get_te_has_grad(self):
        return self.text_encoder.encoder.block[0].layer[0].SelfAttention.q.weight.requires_grad

    def save_model(self, output_path, meta, save_dtype):
        # only save the unet
        transformer: Wan21 = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, 'transformer'),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, 'aitk_meta.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get('noise')
        batch = kwargs.get('batch')
        if batch is None:
            raise ValueError("Batch is not provided")
        if noise is None:
            raise ValueError("Noise is not provided")
        return (noise - batch.latents).detach()

    def convert_lora_weights_before_save(self, state_dict):
        return convert_to_original(state_dict)

    def convert_lora_weights_before_load(self, state_dict):
        return convert_to_diffusers(state_dict)
    
    def get_base_model_version(self):
        return "wan_2.1"
    
    def get_transformer_block_names(self):
        return ['blocks']
