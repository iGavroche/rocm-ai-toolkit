import torch


def get_optimizer(
        params,
        optimizer_type='adam',
        learning_rate=1e-6,
        optimizer_params=None
):
    if optimizer_params is None:
        optimizer_params = {}
    lower_type = optimizer_type.lower()
    if lower_type.startswith("dadaptation"):
        # dadaptation optimizer does not use standard learning rate. 1 is the default value
        import dadaptation
        print("Using DAdaptAdam optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0
        if lower_type.endswith('lion'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type.endswith('adam'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type == 'dadaptation':
            # backwards compatibility
            optimizer = dadaptation.DAdaptAdam(params, eps=1e-6, lr=use_lr, **optimizer_params)
            # warn user that dadaptation is deprecated
            print("WARNING: Dadaptation optimizer type has been changed to DadaptationAdam. Please update your config.")
    elif lower_type.startswith("prodigy8bit"):
        from toolkit.optimizers.prodigy_8bit import Prodigy8bit
        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy8bit(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type.startswith("prodigy"):
        from prodigyopt import Prodigy

        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type == "adam8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
    elif lower_type == "adamw8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
    elif lower_type.endswith("8bit"):
        # Check if ROCm backend is being used (bitsandbytes doesn't fully support ROCm)
        from toolkit.backend_utils import is_rocm_available
        is_rocm = is_rocm_available()
        
        if is_rocm:
            # ROCm backend: use toolkit's own 8-bit implementations or fallback to regular optimizers
            print(f"WARNING: {optimizer_type} requested but bitsandbytes doesn't fully support ROCm.")
            print(f"Falling back to alternative 8-bit implementation or regular optimizer.")
            
            if lower_type == "adam8bit":
                from toolkit.optimizers.adam8bit import Adam8bit
                return Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
            elif lower_type == "adamw8bit":
                from toolkit.optimizers.adam8bit import Adam8bit
                return Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
            elif lower_type == "ademamix8bit":
                # No direct equivalent, fallback to AdamW8
                print("WARNING: AdEMAMix8bit not available for ROCm, using AdamW8 instead")
                from toolkit.optimizers.adam8bit import Adam8bit
                return Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
            elif lower_type == "lion8bit":
                # Fallback to regular Lion if available, otherwise AdamW
                try:
                    from lion_pytorch import Lion
                    print("WARNING: Lion8bit not available for ROCm, using regular Lion instead")
                    return Lion(params, lr=learning_rate, **optimizer_params)
                except ImportError:
                    print("WARNING: Lion8bit not available for ROCm, using AdamW8 instead")
                    from toolkit.optimizers.adam8bit import Adam8bit
                    return Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
            else:
                raise ValueError(f'Unknown optimizer type {optimizer_type}')
        else:
            # CUDA backend: try bitsandbytes
            try:
                import bitsandbytes
                
                if lower_type == "adam8bit":
                    return bitsandbytes.optim.Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
                if lower_type == "ademamix8bit":
                    return bitsandbytes.optim.AdEMAMix8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
                elif lower_type == "adamw8bit":
                    return bitsandbytes.optim.AdamW8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
                elif lower_type == "lion8bit":
                    return bitsandbytes.optim.Lion8bit(params, lr=learning_rate, **optimizer_params)
                else:
                    raise ValueError(f'Unknown optimizer type {optimizer_type}')
            except (ImportError, RuntimeError, AttributeError) as e:
                # bitsandbytes not available or failed to load, fallback to toolkit implementations
                print(f"WARNING: bitsandbytes not available or failed to load: {e}")
                print(f"Falling back to alternative 8-bit implementation for {optimizer_type}")
                
                if lower_type == "adam8bit":
                    from toolkit.optimizers.adam8bit import Adam8bit
                    return Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
                elif lower_type == "adamw8bit":
                    from toolkit.optimizers.adam8bit import Adam8bit
                    return Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
                elif lower_type == "ademamix8bit":
                    from toolkit.optimizers.adam8bit import Adam8bit
                    return Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
                elif lower_type == "lion8bit":
                    try:
                        from lion_pytorch import Lion
                        return Lion(params, lr=learning_rate, **optimizer_params)
                    except ImportError:
                        from toolkit.optimizers.adam8bit import Adam8bit
                        return Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
                else:
                    raise ValueError(f'Unknown optimizer type {optimizer_type}')
    elif lower_type == 'adam':
        optimizer = torch.optim.Adam(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=learning_rate, **optimizer_params)
        except ImportError:
            raise ImportError("Please install lion_pytorch to use Lion optimizer -> pip install lion-pytorch")
    elif lower_type == 'adagrad':
        optimizer = torch.optim.Adagrad(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'adafactor':
        from toolkit.optimizers.adafactor import Adafactor
        if 'relative_step' not in optimizer_params:
            optimizer_params['relative_step'] = False
        if 'scale_parameter' not in optimizer_params:
            optimizer_params['scale_parameter'] = False
        if 'warmup_init' not in optimizer_params:
            optimizer_params['warmup_init'] = False
        optimizer = Adafactor(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic':
        from toolkit.optimizers.automagic import Automagic
        optimizer = Automagic(params, lr=float(learning_rate), **optimizer_params)
    else:
        raise ValueError(f'Unknown optimizer type {optimizer_type}')
    return optimizer
