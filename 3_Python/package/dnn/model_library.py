import sys
import importlib.util
from inspect import getfile
from os import listdir, getcwd
from os.path import isfile, join
from glob import iglob


class ModelRegistry:
    __models_avai: dict = dict()

    def __init__(self):
        """Class for building the overview of neural networks"""
        pass

    def register(self, fn):
        """Adding a class with neural network topology to system"""
        self.__models_avai[fn.__name__] = fn
        return fn

    def build_model(self, name: str, *args, **kwargs):
        """Build the model"""
        return self.__models_avai[name](*args, **kwargs)

    def get_model_library_overview(self, index='', do_print=True) -> list:
        """Getting an overview of existing models in library
        Args:
            index:      Index search for specific model names
            do_print:   Do print the overview
        Return:
            List with all keys functions
        """
        if do_print:
            print("\nOverview of available neural network models"
                  "\n====================================================")
            idx = 0
            for key, func in self.__models_avai.items():
                if index == '' or index in key:
                    print(f"\t#{idx:02d}: {key}")
                    idx += 1
        return [key for key in self.__models_avai.keys()]

    def get_model_library_overview_string(self) -> str:
        string_out = "\nOverview of available neural network models\n==============================================="
        for idx, key in enumerate(self.__models_avai.keys()):
            string_out += f"\n\t#{idx:02d}: {key}"
        return string_out

    def check_model_available(self, model_name: str, do_print=False) -> bool:
        """Function for checking if model name is in Library available (and print where to find)"""
        model_chck = True if model_name in self.__models_avai.keys() else False
        if do_print:
            if model_chck:
                print(f"Model ({model_name} is available at {getfile(self.__models_avai[model_name].build_model(model_name))}")
            else:
                print("Model is not available")
        return model_chck


class ModelLibrary:
    __file2models: list = list()
    __used_key: str = ''

    def __init__(self, key: str = 'models_bib') -> None:
        """Class for searching for used ModelRegistry in package for getting an overview of all models
        Args:
            key:    String with searching module in the python model files
        Return:
            None
        """
        self.__extract_files_search()
        self.__used_key = key

    def __extract_files_search(self, split_path: str = '3_Python') -> None:
        actual_path = getcwd().split(split_path)[0] + '**/models/*.py'
        self.__file2models = [file for file in iglob(actual_path, recursive=True) if isfile(file)]

    def add_external_path_to_models(self, path2folder_models: str) -> None:
        """Function for adding python folder with files of PyTorch models to internal list
        Args:
            path2folder_models:    String to python folder with different models
        Return:
            None
        """
        files_overview = list()
        for file in listdir(path2folder_models):
            files_overview.append(join(path2folder_models, file))
        self.__file2models.extend([file for file in files_overview if isfile(file)])

    def get_registry(self) -> ModelRegistry:
        module = None
        for file in self.__file2models:
            spec = importlib.util.spec_from_file_location('models_bib', file)
            module = importlib.util.module_from_spec(spec)
            sys.modules['models_bib'] = module
            spec.loader.exec_module(module)

        module.models_bib.get_model_library_overview()
        a = module.models_bib.get_model_library_overview()
        b = __import__()
        return module.models_bib


if __name__ == "__main__":
    model_test = ModelLibrary()
    a = model_test.get_registry()
    print(".done")
