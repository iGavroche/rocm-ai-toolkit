import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
# Suppress bitsandbytes errors for ROCm (it will fallback gracefully)
# Set this before any imports that might trigger bitsandbytes
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
import sys
from typing import Union, OrderedDict
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# Suppress bitsandbytes errors for ROCm (it will fallback gracefully)
import warnings
warnings.filterwarnings("ignore", message=".*bitsandbytes.*")
warnings.filterwarnings("ignore", message=".*BNB_BACKEND.*")
warnings.filterwarnings("ignore", message=".*ROCm binary.*")
warnings.filterwarnings("ignore", message=".*Configured.*binary not found.*")

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)
import argparse
from toolkit.print import print_acc, setup_log_to_file

# Initialize accelerator with error handling
try:
    from toolkit.accelerator import get_accelerator
    print_acc("Initializing Accelerator...")
    accelerator = get_accelerator()
    print_acc("Accelerator initialized successfully")
except Exception as e:
    print_acc(f"Error initializing accelerator: {e}")
    import traceback
    print_acc(traceback.format_exc())
    raise

from toolkit.job import get_job


def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true", 
                      help="Quick test mode: reduces steps, batch size, and sample frequency for faster iteration")
    parser.add_argument("--clear-cache", action="store_true",
                      help="Clear Python bytecode cache before running")

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    
    parser.add_argument(
        '--skip-safeguards',
        action='store_true',
        help='Skip system safeguards check (use with caution - can break desktop environment)'
    )
    
    args = parser.parse_args()
    
    # Clear Python bytecode cache if requested
    if args.clear_cache:
        import py_compile
        import shutil
        print_acc("Clearing Python bytecode cache...")
        for root, dirs, files in os.walk('.'):
            # Skip venv and other common directories
            if any(skip in root for skip in ['.venv', 'venv', '__pycache__', '.git', 'node_modules']):
                continue
            if '__pycache__' in dirs:
                cache_dir = os.path.join(root, '__pycache__')
                try:
                    shutil.rmtree(cache_dir)
                    print_acc(f"  Removed {cache_dir}")
                except Exception as e:
                    pass
            for file in files:
                if file.endswith('.pyc'):
                    try:
                        os.remove(os.path.join(root, file))
                    except Exception:
                        pass
        print_acc("Cache cleared!")
    
    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    
    # Set quick test mode in environment if requested
    if args.quick_test:
        os.environ["QUICK_TEST_MODE"] = "1"
        print_acc("Quick test mode enabled: reducing steps, batch size, and sample frequency")
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")
        
        # Set up system safeguards to prevent breaking desktop environment
        if not args.skip_safeguards:
            try:
                from toolkit.system_safeguards import setup_system_safeguards
                # Auto-continue without confirmation (safeguards still check memory, etc.)
                if not setup_system_safeguards(require_confirmation=False):
                    print_acc("Training cancelled due to system safeguards.")
                    sys.exit(1)
            except ImportError as e:
                print_acc(f"Warning: Could not import system safeguards: {e}")
                print_acc("Install psutil: pip install psutil")
                print_acc("Continuing without safeguards (risky on desktop systems)")
            except Exception as e:
                print_acc(f"Warning: Error setting up safeguards: {e}")
                print_acc("Continuing without safeguards (risky on desktop systems)")

    for config_file in config_file_list:
        try:
            print_acc(f"[DEBUG] Loading job from config: {config_file}")
            import sys
            sys.stdout.flush()
            
            print_acc(f"[DEBUG] Calling get_job...")
            sys.stdout.flush()
            job = get_job(config_file, args.name)
            
            print_acc(f"[DEBUG] Job loaded successfully, type: {type(job).__name__}")
            sys.stdout.flush()
            
            print_acc(f"[DEBUG] Starting job.run()...")
            sys.stdout.flush()
            job.run()
            
            print_acc(f"[DEBUG] Job completed successfully")
            sys.stdout.flush()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e


if __name__ == '__main__':
    main()
