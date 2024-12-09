import importlib.util
from os import listdir, getcwd
from os.path import isfile, join
from glob import iglob
from package.dnn.pytorch_handler import ModelRegistry


class ModelLibrary:
    __file2models: list = list()
    __used_key: str = ''
    __used_methods: dict = {}

    def __init__(self, key: str = 'models_available') -> None:
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

    def __extract_methods(self) -> dict:
        methods_overview = dict()

        for file in self.__file2models:
            spec = importlib.util.spec_from_file_location("module.name", file)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)

            if self.__used_key in dir(foo):
                available_modules = foo.models_available.get_model_overview(do_print=False)
                for model in available_modules:
                    methods_overview.update({model: foo.models_available})
        return methods_overview

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

    def get_model_library_overview(self, print_results=False) -> None:
        """Function for getting an overview of all callable PyTorch models in the Repository (only print in terminal)
        Args:
            print_results:  Do print the overview
        Return:
            None
        """
        self.__used_methods = self.__extract_methods()
        if print_results:
            print("\nAvailable methods:"
                  "\n==================================================")
            for idx, model_name in enumerate(self.__used_methods.keys()):
                print(f"#{idx:03d}: {model_name}")

    def get_model_registry(self, attribute: str) -> ModelRegistry:
        """Function for getting the ModelRegistry from Python file with defined PyTorch models
        Args:
            attribute:  String with attribute for given model name
        Return:
            String with model name (Comment: from internal list, the last item is returned)
        """
        self.__used_methods = self.__extract_methods()
        methods_choose = list()
        for key in self.__used_methods:
            if attribute in key:
                methods_choose.append(key)
        return self.__used_methods[methods_choose[-1]]


if __name__ == "__main__":
    model_test = ModelLibrary()
    model_test.get_model_library_overview(True)
    a = model_test.get_model_registry('cnn_ae_v1')

    print(".done")
