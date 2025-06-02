def get_repo_name() -> str:
    """Getting string with repo name"""
    from os.path import dirname
    from pathlib import Path
    import denspp as ref

    path_to_import = dirname(ref.__file__)
    return Path(path_to_import).parts[-1]


def get_path_to_project(new_folder: str='') -> str:
    """Function for getting the path to find the project folder structure in application.
    :param new_folder:  New folder path
    :return:            String of absolute path to start the project structure
    """
    from os.path import dirname, join, abspath
    from pathlib import Path
    import main_pipeline as ref

    path_to_import = dirname(ref.__file__)
    path_split = Path(path_to_import).parts
    path_to_proj = dirname(join(*[path_seg for path_seg in path_split], new_folder, ''))
    return abspath(path_to_proj)


def get_path_to_project_templates() -> str:
    """Getting the path to the project templates"""
    from os.path import dirname, abspath
    import denspp.offline.template as ref

    path_to_temp = dirname(ref.__file__)
    return abspath(path_to_temp)


def get_path_to_project_start(new_folder: str='', folder_ref: str='') -> str:
    """Function for getting the path to find the project folder structure.
    :param new_folder:  New folder path (optional)
    :param folder_ref:  Folder reference for finding project path (best case: repo name, optional)
    :return:            String of absolute path to start the project structure
    """
    from os import getcwd
    from os.path import dirname, join, abspath
    from pathlib import Path

    if get_repo_name() in getcwd() and not folder_ref:
        import denspp as ref
        path_to_import = dirname(ref.__file__)
        path_split = Path(path_to_import).parts[:-1]
        path_to_proj = dirname(join(*[path_seg for path_seg in path_split], ''))
    else:
        path_to_proj = join(getcwd().split(folder_ref)[0], folder_ref) if folder_ref else getcwd()
    path_start = join(path_to_proj, new_folder)
    return abspath(path_start)
