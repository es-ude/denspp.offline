from torch import nn, Tensor, unsqueeze


class dnn_rgc_v1(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self):
        super().__init__()
        self.out_modelname = 'dnn_ae_v1'
        self.out_modeltyp = 'Autoencoder'
        self.model_shape = (1, 32)
        self.model_embedded = False
        iohiddenlayer = [self.model_shape[1], 20, 14, 3]
        do_train_bias = True
        do_train_batch = True

        self.classifier = nn.Sequential(

        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)