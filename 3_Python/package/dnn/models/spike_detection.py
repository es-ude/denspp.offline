from torch import nn, Tensor, unsqueeze


class dnn_sda_v1(nn.Module):
    """Class of a dense-layer based spike detection classifier"""
    def __init__(self):
        super().__init__()
        self.out_modelname = 'dnn_sda_v1'
        self.out_modeltyp = 'Classification'
        self.model_embedded = False
        self.model_shape = (1, 9)
        lin_size = [self.model_shape[1], 6, 4, 1]
        do_train_bias = True
        self.out_class = ['Spike', 'Non-Spike']

        self.detector = nn.Sequential(
            nn.Linear(lin_size[0], lin_size[1]),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.ReLU(),
            nn.Linear(lin_size[1], lin_size[2]),
            nn.BatchNorm1d(lin_size[2], affine=do_train_bias),
            nn.ReLU(),
            nn.Linear(lin_size[2], lin_size[3]),
            nn.BatchNorm1d(lin_size[3], affine=do_train_bias),
            nn.Softmax()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.detector(x)


class cnn_sda_v1(nn.Module):
    """Class of a convolutional spike detection classifier"""
    def __init__(self):
        super().__init__()
        self.out_modelname = 'cnn_sda_v1'
        self.out_modeltyp = 'Classification'
        self.model_embedded = False
        self.model_shape = (1, 9)
        kernel_layer = [1, 4, 3]
        kernel_size = [3, 3]
        kernel_stride = [2, 1]
        kernel_padding = [0, 0, 0]
        pool_size = [2]
        lin_size = [3, 1]
        do_train_bias = True
        self.out_class = ['Spike', 'Non-Spike']

        self.detector = nn.Sequential(
            nn.Conv1d(kernel_layer[0], kernel_layer[1], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm1d(kernel_layer[1], affine=do_train_bias),
            nn.Tanh(),
            nn.Conv1d(kernel_layer[1], kernel_layer[2], kernel_size[1],
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm1d(kernel_layer[2], affine=do_train_bias),
            nn.Tanh(),
            nn.MaxPool1d(pool_size[0]),
            nn.Flatten(start_dim=1),
            nn.Linear(lin_size[0], lin_size[1]),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.Softmax()
        )

    def forward(self, x: Tensor) -> Tensor:
        xin = unsqueeze(x, dim=1)
        return self.detector(xin)
