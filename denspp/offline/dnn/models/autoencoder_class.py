from torch import nn, Tensor, argmax


class synthetic_ae_cl_v1(nn.Module):
    """Classification model of autoencoder output"""
    def __init__(self, input_size=6, output_size=5):
        super().__init__()
        self.model_shape = (1, input_size)
        lin_size = [input_size, 16, 12, output_size]
        lin_drop = [0.0, 0.0]
        do_train_bias = True

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(lin_size[0], lin_size[1]),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(lin_drop[0]),
            nn.Linear(lin_size[1], lin_size[2]),
            nn.BatchNorm1d(lin_size[2], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(lin_drop[1]),
            nn.Linear(lin_size[2], lin_size[3]),
            nn.BatchNorm1d(lin_size[3], affine=do_train_bias),
            # nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        val = self.classifier(x)
        return val, argmax(val, dim=1)
