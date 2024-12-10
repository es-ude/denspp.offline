from torch import nn, Tensor, argmax, flatten
from package.dnn.model_library import ModelRegistry


models_bib = ModelRegistry()


@models_bib.register
class synthetic__cl_v1(nn.Module):
    def __init__(self, input_size=32, output_size=5):
        """DL model for classifying neural spike activity (MLP)"""
        super().__init__()
        self.model_shape = (1, input_size)
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        config_network = [input_size, 40, 32, 20, 12, output_size]

        # --- Model Deployment
        self.model = nn.Sequential()
        for idx, layer_size in enumerate(config_network[1:], start=1):
            self.model.add_module(f"linear_{idx:02d}",
                                  nn.Linear(in_features=config_network[idx - 1], out_features=layer_size,
                                            bias=do_train_bias))
            self.model.add_module(f"batch1d_{idx:02d}",
                                  nn.BatchNorm1d(num_features=layer_size, affine=do_train_batch))
            if not idx == len(config_network) - 1:
                self.model.add_module(f"act_{idx:02d}", nn.ReLU())
            else:
                # self.model.add_module(f"soft", nn.Softmax(dim=1))
                pass

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)
