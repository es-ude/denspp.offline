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


def check_value_range(value: float | int, range: list[float | int]) -> bool:
    """Function for checking if value is within range
    :param value:     Value to check (float or integer)
    :param range:     List with two values to indicate the range
    :return:          Boolean if value is in range
    """
    assert len(range) == 2, "Array should have 2 elements [min, max]"
    return range[0] <= value <= range[1]


def is_close(value: float, target: float, tolerance: float=0.05) -> bool:
    """Function for checking if float value is in near of the target value
    :param value:       Float value to check
    :param target:      Target value
    :param tolerance:   Tolerance value [around target value]
    """
    assert tolerance > 0
    return abs(value - target) <= abs(tolerance)


def get_path_to_project(new_folder: str='', max_levels: int=5) -> str:
    """Function for getting the path to find the project folder structure in application.
    :param new_folder:  New folder path
    :param max_levels:  Max number of levels to get-out for finding pyproject.toml
    :return:            String of absolute path to start the project structure
    """
    from pathlib import Path
    cwd = Path(".").absolute()
    current = cwd

    def is_project_root(p):
        return (p / "pyproject.toml").exists()

    for _ in range(max_levels):
        if is_project_root(current):
            return str(current / new_folder)
        current = current.parent

    if is_project_root(current):
        return str(current / new_folder)
    return str(cwd)


def get_path_to_project_templates() -> str:
    """Getting the path to the project templates
    :return:    String with path"""
    from os.path import dirname, abspath
    import denspp.offline.template as ref

    path_to_temp = dirname(ref.__file__)
    return abspath(path_to_temp)
