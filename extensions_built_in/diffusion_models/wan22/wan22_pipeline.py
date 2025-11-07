
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
            self.transformer.to(device)
            flush()

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
        
        conditioning = None # wan2.2 i2v conditioning
        # check shape of latents to see if it is first frame conditioned for 2.2 14b i2v
        if latents is not None:
            if latents.shape[1] == 36:
                # first 16 channels are latent. other 20 are conditioning
                conditioning = latents[:, 16:]
                latents = latents[:, :16]
                
                # we need to trick the in_channls to think it is only 16 channels
                num_channels_latents = 16
                
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
                
                # DIAGNOSTICS: Log before transformer forward pass
                if i == 0:
                    print(f"\n[DIAGNOSTIC] Starting first inference step (i={i}, t={t})")
                    print(f"[DIAGNOSTIC] Device: {device}, Transformer device: {next(iter(self.transformer.parameters())).device}")
                    if torch.cuda.is_available():
                        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                        print(f"[DIAGNOSTIC] GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved, {mem_total:.2f} GB total")
                    import time
                    step_start_time = time.time()
                
                if boundary_timestep is None or t >= boundary_timestep:
                    if self._aggressive_offload and current_model != self.transformer:
                        if self.transformer_2 is not None:
                            self.transformer_2.to("cpu")
                        self.transformer.to(device)
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    if self._aggressive_offload and current_model != self.transformer_2:
                        if self.transformer is not None:
                            self.transformer.to("cpu")
                        if self.transformer_2 is not None:
                            self.transformer_2.to(device)
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2
                    
                latent_model_input = latents.to(device, transformer_dtype)
                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])
                
                pre_condition_latent_model_input = latent_model_input.clone()
                
                if conditioning is not None:
                    # conditioning is first frame conditioning for 2.2 i2v
                    latent_model_input = torch.cat(
                        [latent_model_input, conditioning], dim=1)

                # DIAGNOSTICS: Log before transformer forward pass
                if i == 0:
                    print(f"[DIAGNOSTIC] About to call transformer forward pass...")
                    print(f"[DIAGNOSTIC] latent_model_input shape: {latent_model_input.shape}, device: {latent_model_input.device}")
                    print(f"[DIAGNOSTIC] prompt_embeds shape: {prompt_embeds.shape}, device: {prompt_embeds.device}")
                    print(f"[DIAGNOSTIC] Transformer type: {type(current_model).__name__}")
                    try:
                        from toolkit.util.quantize import is_model_quantized
                        is_quantized = is_model_quantized(current_model)
                        print(f"[DIAGNOSTIC] Transformer is quantized: {is_quantized}")
                    except:
                        print(f"[DIAGNOSTIC] Could not check quantization status")
                        is_quantized = False
                    import time
                    forward_start = time.time()
                    print(f"[DIAGNOSTIC] Starting transformer forward pass at {forward_start:.2f}...")
                
                # WORKAROUND: ROCm driver crashes (0xC0000005) with quantized models + PYTORCH_NO_HIP_MEMORY_CACHING=1
                # The crash happens in rocblas_create_handle()/miopenCreate() during convolution operations
                # We'll try to work around this by ensuring all operations are properly synchronized
                import os
                use_aggressive_clear = False
                try:
                    from toolkit.util.quantize import is_model_quantized
                    is_quantized = is_model_quantized(current_model)
                    if is_quantized and os.environ.get('PYTORCH_NO_HIP_MEMORY_CACHING') == '1':
                        use_aggressive_clear = True
                        if i == 0:
                            print(f"[WORKAROUND] Transformer is quantized - using aggressive memory clearing and synchronization")
                            print(f"[WORKAROUND] ROCm driver crashes with quantized models + PYTORCH_NO_HIP_MEMORY_CACHING=1")
                            print(f"[WORKAROUND] Attempting GPU forward pass with full synchronization...")
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            # Force a small operation to ensure driver is ready
                            _ = torch.zeros(1, device=device)
                            torch.cuda.synchronize()
                            print(f"[WORKAROUND] Memory cleared and driver synchronized, attempting GPU forward pass...")
                except:
                    pass
                
                # Run on GPU (as requested - we're on ROCm/gfx1151)
                # Note: This may crash the ROCm driver (0xC0000005) in rocblas/miopen with quantized models
                try:
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    # Synchronize immediately after forward pass to catch any driver issues
                    if use_aggressive_clear:
                        torch.cuda.synchronize()
                except RuntimeError as e:
                    if "0xC0000005" in str(e) or "access violation" in str(e).lower():
                        if i == 0:
                            print(f"[ERROR] ROCm driver crashed during transformer forward pass")
                            print(f"[ERROR] This is a known issue with quantized models + PYTORCH_NO_HIP_MEMORY_CACHING=1 on gfx1151")
                            print(f"[ERROR] Consider: 1) Removing PYTORCH_NO_HIP_MEMORY_CACHING=1, 2) Using unquantized models, 3) Updating ROCm drivers")
                        raise
                    else:
                        raise
                
                # DIAGNOSTICS: Log after transformer forward pass
                if i == 0:
                    forward_elapsed = time.time() - forward_start
                    print(f"[DIAGNOSTIC] Transformer forward pass completed in {forward_elapsed:.2f} seconds")
                    print(f"[DIAGNOSTIC] noise_pred shape: {noise_pred.shape}, device: {noise_pred.device}")

                if self.do_classifier_free_guidance:
                    # Clear memory before unconditional forward pass if needed
                    if use_aggressive_clear:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
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
