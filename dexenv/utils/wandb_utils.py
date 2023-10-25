import os

from dexenv.utils.common import pathlib_file


def create_artifact_download_path(artifact_name):
    cwd = pathlib_file(os.getcwd())
    wandb_dir = cwd / pathlib_file(os.environ['WANDB_DIR'])
    wandb_dir = wandb_dir.parent
    artifact_dir = wandb_dir.joinpath('artifacts', f'{artifact_name}')
    return artifact_dir


def create_artifact_name(run_id, eval=False):
    artifact_name = f'train-model-{run_id}' if not eval else f'eval-model-{run_id}'
    return artifact_name


def create_model_name(eval=False):
    model_name = f'train-model.pt' if not eval else f'eval-model.pt'
    return model_name
