
import torch
from toolkit.basic import flush
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import  WanPipeline, WanTransformer3DModel, AutoencoderKLWan
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from typing import List
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.pipelines.wan.pipeline_wan import XLA_AVAILABLE
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput


class Wan22Pipeline(WanPipeline):
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
        aggressive_offload: bool = False,
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
        self._aggressive_offload = aggressive_offload
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
        guidance_scale_2: Optional[float] = None,
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
        noise_mask: Optional[torch.Tensor] = None,
    ):

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
            
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)
        

        width = width // (self.vae.config.scale_factor_spatial * 2) * (self.vae.config.scale_factor_spatial * 2)
        height = height // (self.vae.config.scale_factor_spatial * 2) * (self.vae.config.scale_factor_spatial * 2)

        # unload vae and transformer
        vae_device = self.vae.device
        transformer_device = self.transformer.device
        text_encoder_device = self.text_encoder.device
        device = self._exec_device
        
        if self._aggressive_offload:
            print("Unloading vae")
            self.vae.to("cpu")
            print("Unloading transformer")
            self.transformer.to("cpu")
            if self.transformer_2 is not None:
                self.transformer_2.to("cpu")
            
            # For ROCm, handle text encoder device transfer with error handling
            try:
                from toolkit.backend_utils import is_rocm_available, synchronize_gpu
                is_rocm = is_rocm_available()
            except ImportError:
                is_rocm = False
            
            if is_rocm:
                synchronize_gpu()
                try:
                    self.text_encoder.to(device)
                    synchronize_gpu()
                except (RuntimeError, Exception) as e:
                    error_str = str(e)
                    if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
                        # Keep text encoder on CPU if transfer fails
                        print("Warning: Text encoder device transfer failed on ROCm, keeping on CPU")
                        synchronize_gpu()
                    else:
                        raise
            else:
                self.text_encoder.to(device)
            flush()
        

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2
        )
        
        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
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
        if self._aggressive_offload:
            # unload text encoder
            print("Unloading text encoder")
            self.text_encoder.to("cpu")
            
            # For ROCm, handle transformer device transfer with error handling
            try:
                from toolkit.backend_utils import is_rocm_available, synchronize_gpu
                is_rocm = is_rocm_available()
            except ImportError:
                is_rocm = False
            
            if is_rocm:
                synchronize_gpu()
                try:
                    self.transformer.to(device)
                    synchronize_gpu()
                except Exception as e:
                    # Catch all exceptions including AcceleratorError
                    error_str = str(e)
                    error_type = type(e).__name__
                    if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in error_type or "HIP error" in error_str:
                        # Keep transformer on CPU if transfer fails
                        print("Warning: Transformer device transfer failed on ROCm, keeping on CPU")
                        synchronize_gpu()
                    else:
                        raise
            else:
                self.transformer.to(device)
            flush()

        transformer_dtype = self.transformer.dtype
        # For ROCm, handle prompt_embeds device transfer with error handling
        try:
            from toolkit.backend_utils import is_rocm_available, synchronize_gpu
            is_rocm = is_rocm_available()
        except ImportError:
            is_rocm = False
        
        if is_rocm:
            synchronize_gpu()
            try:
                prompt_embeds = prompt_embeds.to(device, transformer_dtype)
                synchronize_gpu()
            except (RuntimeError, Exception) as e:
                error_str = str(e)
                if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
                    # Keep on CPU if transfer fails - PyTorch will handle cross-device operations
                    pass
                else:
                    raise
        else:
            prompt_embeds = prompt_embeds.to(device, transformer_dtype)
        
        if negative_prompt_embeds is not None:
            if is_rocm:
                synchronize_gpu()
                try:
                    negative_prompt_embeds = negative_prompt_embeds.to(device, transformer_dtype)
                    synchronize_gpu()
                except (RuntimeError, Exception) as e:
                    error_str = str(e)
                    if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
                        # Keep on CPU if transfer fails
                        pass
                    else:
                        raise
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(device, transformer_dtype)

        # 4. Prepare timesteps
        # For ROCm, handle scheduler device transfer with error handling
        # AcceleratorError is a special exception from accelerate library
        if is_rocm:
            synchronize_gpu()
            try:
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                synchronize_gpu()
            except Exception as e:
                # Catch all exceptions including AcceleratorError
                error_str = str(e)
                error_type = type(e).__name__
                if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in error_type or "HIP error" in error_str:
                    # If scheduler can't use GPU device, use CPU instead
                    print("Warning: Scheduler device transfer failed on ROCm, using CPU device")
                    synchronize_gpu()
                    try:
                        self.scheduler.set_timesteps(num_inference_steps, device="cpu")
                        synchronize_gpu()
                    except Exception as e2:
                        print(f"Warning: Scheduler also failed on CPU: {e2}")
                        # Try to continue anyway - scheduler might work
                        synchronize_gpu()
                else:
                    raise
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # Note: If scheduler is on CPU, timesteps will be on CPU too
        # This is fine - PyTorch handles cross-device operations

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        
        conditioning = None # wan2.2 i2v conditioning
        # check shape of latents to see if it is first frame conditioned for 2.2 14b i2v
        if latents is not None:
            if latents.shape[1] == 36:
                # first 16 channels are latent. other 20 are conditioning
                conditioning = latents[:, 16:]
                latents = latents[:, :16]
                
                # we need to trick the in_channls to think it is only 16 channels
                num_channels_latents = 16
                
        # For ROCm, handle prepare_latents with error handling
        # prepare_latents calls torch.randn which can fail on ROCm
        if is_rocm:
            synchronize_gpu()
            try:
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
                synchronize_gpu()
            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in error_type or "HIP error" in error_str:
                    # If GPU latents creation fails, create on CPU and move if possible
                    print("Warning: prepare_latents failed on GPU, trying CPU fallback")
                    synchronize_gpu()
                    try:
                        # If generator is on GPU, we need to handle it carefully
                        # Create a CPU generator if needed to avoid device mismatch
                        cpu_generator = None
                        if generator is not None:
                            try:
                                # Try to get generator state and create CPU version
                                if hasattr(generator, 'device') and generator.device.type == 'cuda':
                                    # Create CPU generator with same seed
                                    cpu_generator = torch.Generator(device='cpu')
                                    if hasattr(generator, 'initial_seed'):
                                        cpu_generator.manual_seed(generator.initial_seed())
                                    else:
                                        cpu_generator.manual_seed(42)  # Fallback seed
                                else:
                                    cpu_generator = generator
                            except Exception:
                                # If we can't copy generator, use None
                                cpu_generator = None
                        
                        # Create latents on CPU first
                        latents = self.prepare_latents(
                            batch_size * num_videos_per_prompt,
                            num_channels_latents,
                            height,
                            width,
                            num_frames,
                            torch.float32,
                            "cpu",
                            cpu_generator,  # Use CPU generator
                            latents,
                        )
                        # Try to move to GPU (but don't fail if it doesn't work)
                        synchronize_gpu()
                        try:
                            latents = latents.to(device)
                            synchronize_gpu()
                        except Exception:
                            # Keep on CPU - will work with cross-device ops
                            synchronize_gpu()
                    except Exception as e2:
                        error_str2 = str(e2)
                        print(f"Warning: prepare_latents CPU fallback failed: {error_str2[:200]}")
                        # Last resort: create simple random tensor on CPU
                        # Use the same shape calculation as prepare_latents would
                        try:
                            # Calculate shape the same way prepare_latents does
                            vae_scale_factor = self.vae_scale_factor_spatial if hasattr(self, 'vae_scale_factor_spatial') else 8
                            vae_scale_factor_temporal = self.vae_scale_factor_temporal if hasattr(self, 'vae_scale_factor_temporal') else 4
                            
                            latent_height = height // (vae_scale_factor * 2) * (vae_scale_factor * 2)
                            latent_width = width // (vae_scale_factor * 2) * (vae_scale_factor * 2)
                            latent_frames = num_frames
                            if latent_frames % vae_scale_factor_temporal != 1:
                                latent_frames = latent_frames // vae_scale_factor_temporal * vae_scale_factor_temporal + 1
                            
                            shape = (
                                batch_size * num_videos_per_prompt,
                                num_channels_latents,
                                latent_height // (vae_scale_factor * 2),
                                latent_width // (vae_scale_factor * 2),
                                latent_frames // vae_scale_factor_temporal,
                            )
                            latents = torch.randn(shape, dtype=torch.float32, device="cpu")
                            print("Warning: Using simple CPU tensor creation as fallback")
                        except Exception as e3:
                            print(f"Error: All fallback methods failed: {e3}")
                            import traceback
                            traceback.print_exc()
                            raise
                else:
                    raise
        else:
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
        
        mask = noise_mask
        if mask is None:
            # For ROCm, handle mask creation with error handling
            if is_rocm:
                synchronize_gpu()
                try:
                    mask = torch.ones(latents.shape, dtype=torch.float32, device=device)
                    synchronize_gpu()
                except Exception as e:
                    error_str = str(e)
                    error_type = type(e).__name__
                    if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in error_type or "HIP error" in error_str:
                        # Create mask on CPU and move if possible
                        synchronize_gpu()
                        mask = torch.ones(latents.shape, dtype=torch.float32, device="cpu")
                        try:
                            mask = mask.to(device)
                            synchronize_gpu()
                        except Exception:
                            # Keep on CPU - will work with cross-device ops
                            synchronize_gpu()
                    else:
                        raise
            else:
                mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - \
            num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None
        
        current_model = self.transformer
        
        if self._aggressive_offload:
            # we don't have one loaded yet in aggressive offload mode
            current_model = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                
                if boundary_timestep is None or t >= boundary_timestep:
                    if self._aggressive_offload and current_model != self.transformer:
                        if self.transformer_2 is not None:
                            self.transformer_2.to("cpu")
                        # For ROCm, handle transformer device transfer with error handling
                        if is_rocm:
                            synchronize_gpu()
                            try:
                                self.transformer.to(device)
                                synchronize_gpu()
                            except (RuntimeError, Exception) as e:
                                error_str = str(e)
                                if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
                                    # Keep on CPU if transfer fails
                                    synchronize_gpu()
                                else:
                                    raise
                        else:
                            self.transformer.to(device)
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    if self._aggressive_offload and current_model != self.transformer_2:
                        if self.transformer is not None:
                            self.transformer.to("cpu")
                        if self.transformer_2 is not None:
                            # For ROCm, handle transformer_2 device transfer with error handling
                            if is_rocm:
                                synchronize_gpu()
                                try:
                                    self.transformer_2.to(device)
                                    synchronize_gpu()
                                except (RuntimeError, Exception) as e:
                                    error_str = str(e)
                                    if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
                                        # Keep on CPU if transfer fails
                                        synchronize_gpu()
                                    else:
                                        raise
                            else:
                                self.transformer_2.to(device)
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2
                    
                # For ROCm, handle latent device transfer with error handling
                if is_rocm:
                    synchronize_gpu()
                    try:
                        latent_model_input = latents.to(device, transformer_dtype)
                        synchronize_gpu()
                    except (RuntimeError, Exception) as e:
                        error_str = str(e)
                        if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
                            # Keep on CPU if transfer fails
                            synchronize_gpu()
                            latent_model_input = latents.to("cpu", transformer_dtype)
                        else:
                            raise
                else:
                    latent_model_input = latents.to(device, transformer_dtype)
                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                    # For ROCm, handle timestep device transfer
                    if is_rocm:
                        synchronize_gpu()
                        try:
                            timestep = timestep.to(device)
                            synchronize_gpu()
                        except (RuntimeError, Exception) as e:
                            error_str = str(e)
                            if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
                                # Keep on CPU if transfer fails
                                synchronize_gpu()
                                timestep = timestep.to("cpu")
                            else:
                                raise
                else:
                    # For ROCm, handle timestep device transfer
                    # t comes from timesteps which might be on CPU if scheduler is on CPU
                    timestep = t.expand(latents.shape[0])
                    if is_rocm:
                        synchronize_gpu()
                        try:
                            timestep = timestep.to(device)
                            synchronize_gpu()
                        except (RuntimeError, Exception) as e:
                            error_str = str(e)
                            if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
                                # Keep on CPU if transfer fails - model will handle cross-device
                                synchronize_gpu()
                                timestep = timestep.to("cpu")
                            else:
                                raise
                    else:
                        timestep = timestep.to(device)
                
                pre_condition_latent_model_input = latent_model_input.clone()
                
                if conditioning is not None:
                    # conditioning is first frame conditioning for 2.2 i2v
                    latent_model_input = torch.cat(
                        [latent_model_input, conditioning], dim=1)

                noise_pred = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + current_guidance_scale * \
                        (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False)[0]
                
                # apply i2v mask
                latents = (pre_condition_latent_model_input * (1 - mask)) + (
                    latents * mask
                )

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

        if self._aggressive_offload:
            # unload transformer
            print("Unloading transformer")
            self.transformer.to("cpu")
            if self.transformer_2 is not None:
                self.transformer_2.to("cpu")
            # load vae
            print("Loading Vae")
            self.vae.to(vae_device)
            flush()

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
        
        # move transformer back to device
        if self._aggressive_offload:
            # print("Moving transformer back to device")
            # self.transformer.to(self._execution_device)
            flush()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
