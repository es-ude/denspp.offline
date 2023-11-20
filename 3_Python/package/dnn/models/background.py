from torch import nn, Tensor, argmax

class sda_bae_v1(nn.Module):
    """Class to identify spike type from NEO operator"""
    def __init__(self, input_size=32, output_size=4):
        super().__init__()
        self.out_modelname = 'sda_bae_v1'
        self.out_modeltyp = 'BAE'
        self.model_embedded = False
        self.model_shape = (1, input_size)
        self.out_class = ['Spike', 'Background', 'Artefact', 'Non-Spike']
        do_train_bias = True
        iohiddenlayer = [input_size, 20, 12, output_size]
        use_bias = False

        # --- Encoder Path
        self.classifier = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=use_bias),
            nn.BatchNorm1d(iohiddenlayer[1], affine=do_train_bias),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=use_bias),
            nn.BatchNorm1d(iohiddenlayer[2], affine=do_train_bias),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=use_bias),
            nn.BatchNorm1d(iohiddenlayer[3], affine=do_train_bias),
            nn.Softmax(dim=0)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        xdist = self.classifier(x)
        return xdist, argmax(xdist)
