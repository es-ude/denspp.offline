from inspect import getfile
from importlib import import_module
from importlib import resources as res
from logging import getLogger
import re


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

    def build(self, name: str, *args, **kwargs):
        """Build the model"""
        return self.__models_avai[name](*args, **kwargs)

    def get_library_overview(self, index: str= '', do_print: bool=True) -> list:
        """Getting an overview of existing and registered modules in library
        Args:
            index:      Index search for specific model names
            do_print:   Do print the overview
        Return:
            List with all keys functions
        """
        if do_print:
            print("\nOverview of modules in library:"
                  "\n====================================================")
            idx = 0
            for key, func in self.__models_avai.items():
                if index == '' or index in key:
                    print(f"\t#{idx:02d}: {key}")
                    idx += 1
        return [key for key in self.__models_avai.keys()]

    def get_library_overview_string(self) -> str:
        string_out = "\nOverview of modules in library\n==============================================="
        for idx, key in enumerate(self.__models_avai.keys()):
            string_out += f"\n\t#{idx:02d}: {key}"
        return string_out

    def check_module_available(self, model_name: str, do_print: bool=False) -> bool:
        """Function for checking if module name is in Library available (and print where to find)"""
        model_chck = True if model_name in self.__models_avai.keys() else False
        if do_print:
            if model_chck:
                print(f"Model ({model_name} is available at {getfile(self.__models_avai[model_name].build(model_name))}")
            else:
                print("Model is not available")
        return model_chck

    def register_package(self, package: str) -> None:
        for resource in res.files(package).iterdir():
            if not resource.name.endswith("__"):
                module_name = f"{package}.{resource.name[:-3]}"
                m = import_module(module_name)
                self._logger.debug(f"importing module from: {module_name}")
                for name in m.__dict__:
                    if re.match(self._regex, name.lower()):
                        self._logger.debug(f"registering module: {name}")
                        item = getattr(m, name)
                        self.register(item)

    def register_packages(self, packages: tuple[str, ...]) -> None:
        for p in packages:
            self.register_package(p)



class ModelLibrary:
    """Class for searching all ModelRegistries in repository to get an overview"""
    def get_registry(self, packages: tuple[str, ...] = ("denspp.offline.dnn.models", "src_dnn.models")) -> ModuleRegistryManager:
        m = ModuleRegistryManager(r".*_v\d+")
        m.register_packages(packages)
        return m


class CellLibrary:
    """Class for searching all CellRegistries in repository to get an overview"""
    def get_registry(self, packages: tuple[str, ...] = ("denspp.offline.dnn.cell_bib", "src_dnn.cell_bib")) -> ModuleRegistryManager:
        m = ModuleRegistryManager(r"resort_\W*")
        m.register_packages(packages)
        return m
