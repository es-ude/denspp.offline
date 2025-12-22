import re
from os.path import join, exists
from inspect import getfile
from importlib import import_module
from importlib import resources as res
from inspect import signature, Signature
from logging import getLogger
from denspp.offline import get_path_to_project


class ModuleRegistryManager:
    __models_avai: dict = dict()

    def __init__(self, regex: str):
        """Class for building a registry of desired type"""
        self._logger = getLogger(__name__)
        self._regex = regex

    def register(self, fn):
        """Adding a class to system"""
        self.__models_avai[fn.__name__] = fn
        return fn

    def build_object(self, name: str) -> object:
        """Returns only the construct of the object
        :param name:    String with name of the object to call
        :return:        Object instance (must be initialized)
        """
        if name not in self.__models_avai:
            raise AttributeError(f"Object {name} not registered")
        return self.__models_avai[name]

    def build(self, name: str, *args, **kwargs):
        """Build the object
        :param name:    String with name of the object to call
        :return:        Object instance (is already initialized)
        """
        if name not in self.__models_avai:
            raise AttributeError(f"Object {name} not registered")
        return self.__models_avai[name](*args, **kwargs)

    def get_signature(self, name: str) -> list:
        """Getting the signature of the object
        :return:    List of input names of object
        """
        return list(signature(self.build_object(name)).parameters.keys())

    def get_library_overview(self, index: str= '', do_print: bool=False) -> list:
        """Getting an overview of existing and registered modules in library
        Args:
            index:      Index search for specific model names
            do_print:   Do print the overview
        Return:
            List with all keys functions
        """
        if do_print:
            self._logger.info("Overview of modules in library:")
            self._logger.info("====================================================")
            idx = 0
            for key, func in self.__models_avai.items():
                if index == '' or index in key:
                    self._logger.info(f"\t#{idx:02d}: {key}")
                    idx += 1
        return [key for key in self.__models_avai.keys()]

    def check_module_available(self, model_name: str, do_print: bool=False) -> bool:
        """Function for checking if module name is in Library available (and print where to find)"""
        model_chck = True if model_name in self.__models_avai.keys() else False
        if do_print:
            if model_chck:
                self._logger.info(f"Model ({model_name} is available at {getfile(self.__models_avai[model_name].build(model_name))}")
            else:
                self._logger.info("Model is not available")
        return model_chck

    def register_package(self, package: str) -> None:
        overview_data = res.files(package).iterdir()
        for resource in overview_data:
            if not resource.name.endswith("__") and not resource.name.startswith(".") and resource.is_file() and resource.name.endswith(".py"):
                module_name = f"{package}.{resource.name.split(resource.suffix)[0]}"
                m = import_module(module_name)
                self._logger.debug(f"importing module from: {module_name}")
                for name in m.__dict__:
                    self._logger.debug(f"available module: {name}")
                    if re.fullmatch(self._regex, name):
                        self._logger.debug(f"registering module: {name}")
                        item = getattr(m, name)
                        self.register(item)

    def register_packages(self, packages: tuple[str, ...]) -> None:
        for p in packages:
            self.register_package(p)


class DatasetLoaderLibrary:
    """Class for searching all DatasetLoader in repository to get an overview"""
    def get_registry(self, package: str="src_dnn") -> ModuleRegistryManager:
        m = ModuleRegistryManager(r"\bDatasetLoader(Test)?\b")
        chck = exists(join(get_path_to_project(), package))
        m.register_package(package) if chck else m.register_package("denspp.offline.template")
        return m


class ModelLibrary:
    """Class for searching all ModelRegistries in repository to get an overview"""
    def get_registry(self, package: str="src_dnn.models") -> ModuleRegistryManager:
        m = ModuleRegistryManager(r".*_v\d+")
        chck = exists(join(get_path_to_project(), 'src_dnn/models'))
        m.register_packages(("denspp.offline.dnn.models", package)) if chck else m.register_package("denspp.offline.dnn.models")
        return m


class CellLibrary:
    """Class for searching all CellRegistries in repository to get an overview"""
    def get_registry(self, package: str="src_dnn.cell_bib") -> ModuleRegistryManager:
        m = ModuleRegistryManager(r"resort_\W*")
        chck = exists(join(get_path_to_project(), 'src_dnn/cell_bib'))
        m.register_packages(("denspp.offline.dnn.cell_bib", package)) if chck else m.register_package("denspp.offline.dnn.cell_bib")
        return m
