from typing import Union, OrderedDict

from toolkit.config import get_config


def get_job(
        config_path: Union[str, dict, OrderedDict],
        name=None
):
    print(f"[DEBUG] get_job called with config_path: {config_path}, name: {name}")
    config = get_config(config_path, name)
    print(f"[DEBUG] Config loaded, job type: {config.get('job', 'NOT FOUND')}")
    
    if not config['job']:
        raise ValueError('config file is invalid. Missing "job" key')

    job = config['job']
    print(f"[DEBUG] Creating job instance for type: {job}")
    
    if job == 'extract':
        from jobs import ExtractJob
        return ExtractJob(config)
    if job == 'train':
        from jobs import TrainJob
        return TrainJob(config)
    if job == 'mod':
        from jobs import ModJob
        return ModJob(config)
    if job == 'generate':
        from jobs import GenerateJob
        return GenerateJob(config)
    if job == 'extension':
        from jobs import ExtensionJob
        print(f"[DEBUG] Creating ExtensionJob...")
        job_instance = ExtensionJob(config)
        print(f"[DEBUG] ExtensionJob created successfully")
        return job_instance

    # elif job == 'train':
    #     from jobs import TrainJob
    #     return TrainJob(config)
    else:
        raise ValueError(f'Unknown job type {job}')


def run_job(
        config: Union[str, dict, OrderedDict],
        name=None
):
    job = get_job(config, name)
    job.run()
    job.cleanup()
