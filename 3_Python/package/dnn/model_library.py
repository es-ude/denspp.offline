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
    """Class for searching for used ModelRegistry in package for getting an overview of all models"""
    def get_registry(self, packages: tuple[str, ...] = ("package.dnn.models", "src_dnn.models")) -> ModelRegistry:
        m = ModelRegistry()
        m.register_packages(packages)
        return m


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    model_test = ModelLibrary()
    model_test.get_registry().get_model_library_overview()

    print(".done")
