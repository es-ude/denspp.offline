import logging
from pathlib import Path
from shutil import copy

from denspp.offline import get_path_to_project, get_path_to_project_templates
from denspp.offline.data_call.owncloud_handler import OwnCloudDownloader

logger = logging.getLogger(__name__)


def copy_template_files(copy_files: dict, path2start: Path) -> None:
    """Function for copying template files to new folder.
    :param copy_files:          Dictionary of file paths to copy
    :param path2start:          Path to start folder
    :return:                    None
    """
    path2temp = get_path_to_project_templates()
    for file_name, folder_name in copy_files.items():
        src = path2temp / file_name
        dst = path2start / folder_name
        dst.mkdir(exist_ok=True, parents=True)
        if not (dst / file_name).exists():
            copy(src=src, dst=dst)
            logger.debug(f"Copy file from: {src} - to: {dst}")


def init_project_folder(new_folder: str = "") -> None:
    """Generating folder structure in first run
    :param new_folder:      Name of the new folder to create (test case)
    :return:                None
    """
    OwnCloudDownloader()
    folder_structure = ["data", "dataset", "runs", "config", "src", "src_pipe"]
    copy_files = {
        ".gitignore": "",
        "README.md": "",
        "run_tests.py": "",
        "run_pipeline.py": "",
        "call_data.py": "src_pipe",
        "pipeline_plot.py": "src_pipe",
        "pipeline_v0.py": "src_pipe",
    }
    path2start = get_path_to_project(new_folder)
    path2start.mkdir(parents=True, exist_ok=True)

    for folder_name in folder_structure:
        (path2start / folder_name).mkdir(parents=True, exist_ok=True)
        if not (path2start / folder_name).exists():
            logger.debug(f"Creating template folder: {folder_name}")

    copy_template_files(copy_files=copy_files, path2start=path2start)
    init_dnn_folder(new_folder=new_folder)


def init_dnn_folder(new_folder: str = "") -> None:
    """Generating a handler dummy for training neural networks
    :param new_folder:      Name of the new folder to create (test case)
    :return:                None
    """
    folder_start = "src_dnn"
    folder_structure = ["models", "dataset"]
    copy_files = {
        "run_training.py": "",
        "call_dataset.py": folder_start,
        "example_model.py": f"{folder_start}/models",
    }

    # --- Generation process
    path2start = get_path_to_project(new_folder)
    for folder_name in folder_structure:
        (path2start / folder_start / folder_name).mkdir(parents=True, exist_ok=True)
        if not path2start.exists():
            logger.debug(f"Creating template folder: {folder_name}")

    copy_template_files(
        copy_files=copy_files,
        path2start=path2start
    )
