pytest_plugins = ["elasticai.creator.testing"]


def pytest_sessionstart(session):
    from denspp.offline import get_path_to_project
    from shutil import rmtree

    rmtree(get_path_to_project("config"), ignore_errors=True)
    rmtree(get_path_to_project("dataset"), ignore_errors=True)
    rmtree(get_path_to_project("runs"), ignore_errors=True)
    rmtree(get_path_to_project("src_dnn"), ignore_errors=True)
    rmtree(get_path_to_project("temp_test"), ignore_errors=True)


def pytest_sessionfinish(session, exitstatus):
    from denspp.offline import get_path_to_project
    from shutil import rmtree

    rmtree(get_path_to_project("config"), ignore_errors=True)
    rmtree(get_path_to_project("dataset"), ignore_errors=True)
    rmtree(get_path_to_project("src_dnn"), ignore_errors=True)
    rmtree(get_path_to_project("temp_test"), ignore_errors=True)
