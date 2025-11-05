import torch
from toolkit.models.wan21.wan_utils import add_first_frame_conditioning
from toolkit.prompt_utils import PromptEmbeds
from PIL import Image
import torch
from toolkit.config_modules import GenerateImageConfig
from .wan22_pipeline import Wan22Pipeline

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from diffusers import WanImageToVideoPipeline
from torchvision.transforms import functional as TF

from .wan22_14b_model import Wan2214bModel, boundary_ratio_i2v, DualWanTransformer3DModel
from typing import List
from toolkit.config_modules import ModelConfig

class Wan2214bI2VModel(Wan2214bModel):
    arch = "wan22_14b_i2v"
    
    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            model_config=model_config,
            dtype=dtype,
            custom_pipeline=custom_pipeline,
            noise_scheduler=noise_scheduler,
            **kwargs,
        )
        # Override boundary ratio for i2v (0.9 instead of 0.875 for t2v)
        self.multistage_boundaries: List[float] = [boundary_ratio_i2v, 0.0]
    
    def load_wan_transformer(self, transformer_path, subfolder=None):
        """Override to use i2v boundary ratio when creating DualWanTransformer3DModel"""
        # Call parent to load transformers, but we need to override the boundary_ratio
        # Since parent hardcodes boundary_ratio_t2v, we'll patch it after creation
        import os
        from diffusers import WanTransformer3DModel
        from toolkit.util.quantize import quantize_model
        from toolkit.memory_management import MemoryManager
        from toolkit.basic import flush
        from toolkit.backend_utils import is_rocm_available, synchronize_gpu
        
        if self.model_config.split_model_over_gpus:
            raise ValueError("Splitting model over gpus is not supported for Wan2.2 models")
        
        # Get transformer paths (same as parent)
        transformer_path_1 = transformer_path
        subfolder_1 = subfolder
        transformer_path_2 = transformer_path
        subfolder_2 = subfolder
        
        if subfolder_2 is None:
            transformer_path_2 = os.path.join(os.path.dirname(transformer_path_1), "transformer_2")
        else:
            subfolder_2 = "transformer_2"
        
        # Load transformers (reuse parent logic but we'll use i2v boundary)
        self.print_and_status_update("Loading transformer 1")
        dtype = self.torch_dtype
        
        is_rocm = is_rocm_available()
        if is_rocm:
            synchronize_gpu()
        
        transformer_1 = WanTransformer3DModel.from_pretrained(
            transformer_path_1, subfolder=subfolder_1, torch_dtype=dtype,
        )
        
        if is_rocm and transformer_1.dtype != dtype:
            try:
                transformer_1 = transformer_1.to(dtype=dtype)
            except (RuntimeError, Exception) as e:
                if "HIP" in str(e) or "hipError" in str(e):
                    self.print_and_status_update(f"Warning: dtype conversion failed on ROCm: {e}")
                else:
                    raise
        else:
            transformer_1 = transformer_1.to(dtype=dtype)
        
        flush()
        
        if self.model_config.low_vram:
            transformer_1.to('cpu', dtype=dtype)
        else:
            if is_rocm:
                transformer_1 = transformer_1.cpu()
                if dtype != transformer_1.dtype:
                    transformer_1 = transformer_1.to(dtype=dtype)
            else:
                transformer_1.to(self.device_torch, dtype=dtype)
            flush()
        
        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            self.print_and_status_update("Quantizing Transformer 1")
            training_folder = getattr(self, 'training_folder', None)
            quantize_model(self, transformer_1, model_path=transformer_path_1, model_name="transformer_1", training_folder=training_folder)
            flush()
        
        if self.model_config.low_vram or is_rocm:
            transformer_1.to("cpu")
        else:
            transformer_1.to(self.device_torch)
        
        # Load transformer 2 (similar logic)
        self.print_and_status_update("Loading transformer 2")
        if is_rocm:
            synchronize_gpu()
        
        transformer_2 = WanTransformer3DModel.from_pretrained(
            transformer_path_2, subfolder=subfolder_2, torch_dtype=dtype,
        )
        
        if is_rocm and transformer_2.dtype != dtype:
            try:
                transformer_2 = transformer_2.to(dtype=dtype)
            except (RuntimeError, Exception) as e:
                if "HIP" in str(e) or "hipError" in str(e):
                    self.print_and_status_update(f"Warning: dtype conversion failed on ROCm: {e}")
                else:
                    raise
        else:
            transformer_2 = transformer_2.to(dtype=dtype)
        
        flush()
        
        if self.model_config.low_vram:
            transformer_2.to('cpu', dtype=dtype)
        else:
            if is_rocm:
                transformer_2 = transformer_2.cpu()
                if dtype != transformer_2.dtype:
                    transformer_2 = transformer_2.to(dtype=dtype)
            else:
                transformer_2.to(self.device_torch, dtype=dtype)
            flush()
        
        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            self.print_and_status_update("Quantizing Transformer 2")
            training_folder = getattr(self, 'training_folder', None)
            quantize_model(self, transformer_2, model_path=transformer_path_2, model_name="transformer_2", training_folder=training_folder)
            flush()
        
        if self.model_config.low_vram or is_rocm:
            transformer_2.to("cpu")
        else:
            transformer_2.to(self.device_torch)
        
        # Create DualWanTransformer3DModel with i2v boundary ratio
        layer_offloading_transformer = self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0
        self.print_and_status_update("Creating DualWanTransformer3DModel (i2v)")
        transformer = DualWanTransformer3DModel(
            transformer_1=transformer_1,
            transformer_2=transformer_2,
            torch_dtype=self.torch_dtype,
            device=self.device_torch,
            boundary_ratio=boundary_ratio_i2v,  # Use i2v boundary ratio instead of t2v
            low_vram=self.model_config.low_vram,
        )
        
        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is not None:
            self.print_and_status_update("Applying Accuracy Recovery Adapter to Transformers")
            quantize_model(self, transformer)
            flush()
        
        if layer_offloading_transformer:
            MemoryManager.attach(
                transformer_1, self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[transformer_1.scale_shift_table] + [block.scale_shift_table for block in transformer_1.blocks]
            )
            MemoryManager.attach(
                transformer_2, self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[transformer_2.scale_shift_table] + [block.scale_shift_table for block in transformer_2.blocks]
            )
        
        return transformer
    
    def get_generation_pipeline(self):
        """Override to use i2v boundary ratio in pipeline"""
        from diffusers import UniPCMultistepScheduler
        from .wan22_pipeline import Wan22Pipeline
        from .wan22_14b_model import scheduler_configUniPC
        
        scheduler = UniPCMultistepScheduler(**scheduler_configUniPC)
        pipeline = Wan22Pipeline(
            vae=self.vae,
            transformer=self.model.transformer_1,
            transformer_2=self.model.transformer_2,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=scheduler,
            expand_timesteps=self._wan_expand_timesteps,
            device=self.device_torch,
            aggressive_offload=self.model_config.low_vram,
            boundary_ratio=boundary_ratio_i2v,  # Use i2v boundary ratio
        )
        
        return pipeline
    
    
    def generate_single_image(
        self,
        pipeline: Wan22Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        
        # todo 
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)

        num_frames = (
            (gen_config.num_frames - 1) // 4
        ) * 4 + 1  # make sure it is divisible by 4 + 1
        gen_config.num_frames = num_frames

        height = gen_config.height
        width = gen_config.width
        first_frame_n1p1 = None
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img).convert("RGB")

            d = self.get_bucket_divisibility()

            # make sure they are divisible by d
            height = height // d * d
            width = width // d * d

            # resize the control image
            control_img = control_img.resize((width, height), Image.LANCZOS)

            # 5. Prepare latent variables
            # num_channels_latents = self.transformer.config.in_channels
            num_channels_latents = 16
            latents = pipeline.prepare_latents(
                1,
                num_channels_latents,
                height,
                width,
                gen_config.num_frames,
                torch.float32,
                self.device_torch,
                generator,
                None,
            ).to(self.torch_dtype)

            first_frame_n1p1 = (
                TF.to_tensor(control_img)
                .unsqueeze(0)
                .to(self.device_torch, dtype=self.torch_dtype)
                * 2.0
                - 1.0
            )  # normalize to [-1, 1]
            
            # Add conditioning using the standalone function
            gen_config.latents = add_first_frame_conditioning(
                latent_model_input=latents,
                first_frame=first_frame_n1p1,
                vae=self.vae
            )

        # For ROCm, handle device transfer with error handling
        try:
            from toolkit.backend_utils import is_rocm_available, synchronize_gpu
            is_rocm = is_rocm_available()
        except ImportError:
            is_rocm = False
        
        # Use PromptEmbeds.to() method which has ROCm error handling
        if is_rocm:
            synchronize_gpu()
            try:
                conditional_embeds = conditional_embeds.to(self.device_torch, dtype=self.torch_dtype)
                unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=self.torch_dtype)
                synchronize_gpu()
            except (RuntimeError, Exception) as e:
                error_str = str(e)
                if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
                    # Keep on CPU if transfer fails
                    pass
                else:
                    raise
        else:
            conditional_embeds = conditional_embeds.to(self.device_torch, dtype=self.torch_dtype)
            unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=self.torch_dtype)
        
        output = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            height=height,
            width=width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="pil",
            **extra,
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
        batch: DataLoaderBatchDTO,
        **kwargs
    ):
        # videos come in (bs, num_frames, channels, height, width)
        # images come in (bs, channels, height, width)
        with torch.no_grad():
            frames = batch.tensor
            if len(frames.shape) == 4:
                first_frames = frames
            elif len(frames.shape) == 5:
                first_frames = frames[:, 0]
            else:
                raise ValueError(f"Unknown frame shape {frames.shape}")
            
            # Add conditioning using the standalone function
            conditioned_latent = add_first_frame_conditioning(
                latent_model_input=latent_model_input,
                first_frame=first_frames,
                vae=self.vae
            )
        
        noise_pred = self.model(
            hidden_states=conditioned_latent,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred