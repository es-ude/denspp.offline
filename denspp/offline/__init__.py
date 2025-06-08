def check_key_elements(key: str, elements: list[str]) -> bool:
    """Function for checking if all elements are in key (logical AND)
    :param key:         Key to check
    :param elements:    List of elements to check if available in key
    :return:            True if all elements are present in key
    """
    return any(elem == key for elem in elements)


def check_string_equal_elements_all(text: str, elements: list[str]) -> bool:
    """Function for checking if all elements are in text string (logical AND)
    :param text :       String with a text
    :param elements:    List of elements to check if available in text
    :return:            True if all elements are present in text
    """
    return all(elem in text for elem in elements)


def check_string_equal_elements_any(text: str, elements: list[str]) -> bool:
    """Function for checking if elements are in text string (logical OR)
    :param text:        String with a text
    :param elements:    List of elements to check if available in text
    :return:            True if any elements are present in text
    """
    val = any(elem in text for elem in elements)
    return  val


def check_keylist_elements_all(keylist: list[str], elements: list[str]) -> bool:
    """Function for checking if all elements are in key list (logical AND)
    :param keylist:     List with keys to check
    :param elements:    List with elements to check if available in key
    :return:            True if all elements are present in key
    """
    return all(elem in keylist for elem in elements) if len(keylist) else True


def check_keylist_elements_any(keylist: list[str], elements: list[str]) -> bool:
    """Function for checking if all elements are in key list (logical OR)
    :param keylist:     List with keys to check
    :param elements:    List with elements to check if available in key
    :return:            True if any elements are present in key
    """
    return any(elem in keylist for elem in elements) if len(keylist) else True


def check_elem_unique(elements: list) -> bool:
    """Function for checking if all elements are unique
    :param elements:    List of elements to check
    :return:            True if all elements are unique
    """
    from collections import Counter
    from itertools import chain
    chck = elements if not type(elements[0]) == list else list(chain.from_iterable(elements))
    return all(cnt == 1 for cnt in Counter(chck).values())


def is_close(value: float, target: float, tolerance: float=0.05) -> bool:
    """Function for checking if float value is in near of the target value
    :param value:       Float value to check
    :param target:      Target value
    :param tolerance:   Tolerance value [around target value]
    """
    assert tolerance > 0
    return abs(value - target) <= abs(tolerance)


def get_repo_name() -> str:
    """Getting string with repo name"""
    from os.path import dirname
    from pathlib import Path
    import denspp.offline as ref

    path_to_import = dirname(ref.__file__)
    path = Path(path_to_import).parts[-2]
    return path


def get_path_to_project(new_folder: str='') -> str:
    """Function for getting the path to find the project folder structure in application.
    :param new_folder:  New folder path
    :return:            String of absolute path to start the project structure
    """
    from os.path import dirname, join, abspath, exists
    from pathlib import Path
    if exists('main_pipeline.py'):
        import main_pipeline as ref
        path_to_import = dirname(ref.__file__)
    else:
        path_to_import = get_path_to_project_start(new_folder=new_folder)

    path_to_proj = dirname(join(*[path_seg for path_seg in Path(path_to_import).parts], new_folder, ''))
    return abspath(path_to_proj)


def get_path_to_project_templates() -> str:
    """Getting the path to the project templates
    :return:    String with path"""
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
        import denspp.offline as ref
        path_to_import = dirname(ref.__file__)
        path_to_proj = dirname(join(*[path_seg for path_seg in Path(path_to_import).parts[:-2]], ''))
    else:
        path_to_proj = join(getcwd().split(folder_ref)[0], folder_ref) if folder_ref else getcwd()
    path_start = join(path_to_proj, new_folder)
    return abspath(path_start)
