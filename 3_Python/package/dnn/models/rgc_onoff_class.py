from torch import nn, Tensor, unsqueeze, argmax


class dnn_rgc_v1(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=40, output_size=5):
        super().__init__()
        self.out_modelname = 'rgc_class_v1'
        self.out_modeltyp = 'Classification'
        self.model_shape = (1, input_size)
        self.model_embedded = False
        lin_size = [input_size, 50, 26, 18, output_size]
        do_train_bias = True

        self.classifier = nn.Sequential(
            nn.Linear(lin_size[0], lin_size[1]),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.Tanh(),
            nn.Linear(lin_size[1], lin_size[2]),
            nn.BatchNorm1d(lin_size[2], affine=do_train_bias),
            nn.Tanh(),
            nn.Linear(lin_size[2], lin_size[3]),
            nn.BatchNorm1d(lin_size[3], affine=do_train_bias),
            nn.Tanh(),
            nn.Linear(lin_size[3], lin_size[4]),
            nn.BatchNorm1d(lin_size[4], affine=do_train_bias),
            nn.Softmax()
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        val = self.classifier(x)
        dec = argmax(val, dim=1)
        return val, dec

