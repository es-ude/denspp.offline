"""from torch import nn, Tensor, argmax

class rgc_ae_cl_v1(nn.Module):
    ""Classification model of autoencoder output""
    def __init__(self, input_size=6, output_size=5, num_layer):
        super().__init__()
        self.out_modelname = 'ae_class_v1'
        self.out_modeltyp = 'Classification'
        self.model_shape = (1, input_size)
        self.model_embedded = False
        lin_size = [input_size, 40, 32, 24, 16, 12, output_size]
        lin_drop = [0.0, 0.0, 0.0, 0.0, 0.0]
        do_train_bias = True

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(lin_size[0], lin_size[1]),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.SiLU(),
            nn.Dropout(lin_drop[0]),
            nn.Linear(lin_size[1], lin_size[2]),
            nn.BatchNorm1d(lin_size[2], affine=do_train_bias),
            nn.SiLU(),
            nn.Dropout(lin_drop[1]),
            nn.Linear(lin_size[2], lin_size[3]),
            nn.BatchNorm1d(lin_size[3], affine=do_train_bias),
            if num_layer > 2:
                nn.SiLU(),
                nn.Dropout(lin_drop[2]),
                nn.Linear(lin_size[3], lin_size[4]),
                nn.BatchNorm1d(lin_size[4], affine=do_train_bias),
                if num_layer <= 3:
                    nn.Softmax(dim=1)
                else:
                    nn.SiLU(),
                    nn.Dropout(lin_drop[3]),
                    nn.Linear(lin_size[4], lin_size[5]),
                    nn.BatchNorm1d(lin_size[5], affine=do_train_bias),
                    if num_layer > 3:
                        nn.SiLU(),
                        nn.Dropout(lin_drop[4]),
                        nn.Linear(lin_size[5], lin_size[6]),
                        nn.BatchNorm1d(lin_size[6], affine=do_train_bias)
            )


    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        val = self.classifier(x)
        return val, argmax(val, dim=1)"""
