import torch
from torch import nn, Tensor, argmax
from package.dnn.pytorch_handler import __model_settings_common, ModelRegistry


models_available = ModelRegistry()


@models_available.register
class cnn_lstm_dec_v4(__model_settings_common):
    """Class of a convolutional Decoding for feature extraction"""
    def __init__(self, num_clusters, input_samples, output_samples=3):
        super().__init__('CNN+LSTM')
        self.model_shape = (1, num_clusters, 10, 10, input_samples)
        self.model_embedded = False
        # --- Settings of model
        do_bias_train = True
        kernel_layer = [100, 50]
        dense_layer_size = [1000, 720]

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=kernel_layer[0],
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=kernel_layer[0]),
            nn.ReLU(),

            nn.Conv2d(in_channels=kernel_layer[0],
                      out_channels=kernel_layer[1],
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=kernel_layer[1]),
            nn.ReLU(),
        )

        self.dnn_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=dense_layer_size[0],
                out_features=dense_layer_size[1],
                bias=do_bias_train
            ),
            nn.Dropout(0.2),
            nn.BatchNorm1d(dense_layer_size[1]),
            nn.ReLU(),
            nn.Linear(
                in_features=dense_layer_size[1],
                out_features=output_samples,
                bias=do_bias_train
            ),
            nn.Dropout(0.2),
            nn.BatchNorm1d(output_samples),
            nn.Softmax(dim=1)
        )

        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(
            input_size=1800,  # Adjust based on the output size from CNN
            hidden_size=1000,  # Example hidden size, adjust as needed
            num_layers=input_samples,  # Example number of LSTM layers, adjust as needed
            batch_first=True  # Input and output tensors are provided as (batch, seq, feature)
        )

    def forward(self, x):
        batch_size, num_clusters, height, width, num_time_windows = x.shape
        video = []
        for i in range(num_time_windows):
            img = x[:, :, :, :, i]
            img = img.view(batch_size, 2, 10, 10)
            cnn_out = self.cnn_1(img)
            cnn_out = cnn_out.view(batch_size, -1)
            video.append(cnn_out.unsqueeze(1))
        lstm_input = torch.cat(video, dim=1)
        lstm_output, _ = self.lstm(lstm_input)
        pred_con = self.dnn_1(lstm_output[:, -1, :])
        return pred_con, argmax(pred_con, 1)


@models_available.register
class cnn_lstm_dec_v3(__model_settings_common):
    """Class of a convolutional Decoding for feature extraction"""

    def __init__(self, num_clusters=1, input_samples=12, output_samples=3):
        super().__init__('CNN+LSTRM')
        self.model_shape = (1, num_clusters, 10, 10, input_samples)
        self.model_embedded = False
        # --- Settings of model
        do_bias_train = True
        kernel_layer = [num_clusters, 100, 50]
        dense_layer_size = [1000, 720]

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=kernel_layer[1],
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=kernel_layer[1]),
            nn.ReLU(),

            nn.Conv2d(in_channels=kernel_layer[1],
                      out_channels=kernel_layer[2],
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=kernel_layer[2]),
            nn.ReLU(),
        )

        self.dnn_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=dense_layer_size[0],
                out_features=dense_layer_size[1],
                bias=do_bias_train
            ),
            nn.Dropout(0.2),
            nn.BatchNorm1d(dense_layer_size[1]),
            nn.ReLU(),
            nn.Linear(
                in_features=dense_layer_size[1],
                out_features=output_samples,
                bias=do_bias_train
            ),
            nn.Dropout(0.2),
            nn.BatchNorm1d(output_samples),
            nn.Softmax(dim=1)
        )

        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(
            input_size=1800,
            hidden_size=1000,
            num_layers=input_samples,
            batch_first=True
        )

    def forward(self, x):
        batch_size, num_clusters, height, width, num_time_windows = x.shape
        video = []
        for i in range(num_time_windows):
            img = x[:, :, :, :, i]
            img.view(batch_size, num_clusters, 10, 10)
            cnn_out = self.cnn_1(img)
            cnn_out = cnn_out.view(batch_size, -1)
            video.append(cnn_out.unsqueeze(1))
        lstm_input = torch.cat(video, dim=1)
        lstm_output, _ = self.lstm(lstm_input)
        pred_con = self.dnn_1(lstm_output[:, -1, :])
        return pred_con, argmax(pred_con, 1)


@models_available.register
class cnn_lstm_dec_v2(__model_settings_common):
    """Class of a convolutional Decoding for feature extraction"""
    def __init__(self, num_clusters=1, input_samples=12, output_samples=3):
        super().__init__('CNN+LSTM')
        self.model_shape = (1, num_clusters, 10, 10, input_samples)
        self.model_embedded = False
        # --- Settings of model
        do_bias_train = True
        kernel_layer = [num_clusters, 10, 20]
        dense_layer_size = [64, 32]

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(in_channels = num_clusters,
                      out_channels= kernel_layer[1],
                      kernel_size=(3,3),
                      stride= 1,
                      padding= 0),
            nn.BatchNorm2d(num_features= kernel_layer[1]),
            nn.ReLU(),
        )

        self.dnn_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features= dense_layer_size[0],
                out_features= dense_layer_size[1],
                bias=do_bias_train
            ),
            nn.BatchNorm1d(dense_layer_size[1]),
            nn.ReLU(),
            nn.Linear(
                in_features= dense_layer_size[1],
                out_features= output_samples,
                bias=do_bias_train
            ),
            nn.BatchNorm1d(output_samples),
            nn.Softmax()
        )

        self.lstm = nn.LSTM(
            input_size=640,  # Adjust based on the output size from CNN
            hidden_size=64,          # Example hidden size, adjust as needed
            num_layers=input_samples,            # Example number of LSTM layers, adjust as needed
            batch_first=True         # Input and output tensors are provided as (batch, seq, feature)
        )

    def forward(self, x):
        batch_size, num_clusters, height, width, num_time_windows = x.shape
        video = []
        for i in range(num_time_windows):
            img = x[:, :, :, :, i]
            img.view(batch_size, num_clusters, 10, 10)
            cnn_out = self.cnn_1(img)
            cnn_out = cnn_out.view(batch_size, -1)
            video.append(cnn_out.unsqueeze(1))
        lstm_input = torch.cat(video, dim=1)
        lstm_output, _ = self.lstm(lstm_input)
        pred_con = self.dnn_1(lstm_output[:,-1,:])
        return pred_con, argmax(pred_con, 1)


@models_available.register
class test_model_if_pipeline_running(__model_settings_common):
    """Class of a convolutional Decoding for feature extraction but with 3D CNN. Project WiSe 23/24"""

    def __init__(self, num_clusters=1, input_samples=12, output_samples=3):
        super().__init__('CNN+LSTRM')
        self.model_shape = (1, num_clusters, 10, 10, input_samples)
        self.model_embedded = False
        # --- Settings for CNN
        do_bias_train = True
        kernel_layer = [num_clusters, 10, 20]
        kernel_stride = [3, 3, 2] # how much we move
        kernel_padding = [0, 0, 0]
        # --- Settings for DNN/LSTM
        dense_layer_size = [40, 32, output_samples]

        self.cnn_1 = nn.Sequential(
            nn.Conv3d(kernel_layer[0], kernel_layer[1], kernel_size=(3, 3, 1),
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm3d(kernel_layer[1]),
            nn.ReLU(),
            nn.Conv3d(kernel_layer[1], kernel_layer[2], kernel_size=(3, 3, 1),
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm3d(kernel_layer[2]),
            nn.ReLU()
        )
        self.dnn_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dense_layer_size[0], dense_layer_size[1], bias=do_bias_train),
            nn.BatchNorm1d(dense_layer_size[1]),
            nn.ReLU(),
            nn.Linear(dense_layer_size[1], dense_layer_size[2], bias=do_bias_train),
            nn.BatchNorm1d(dense_layer_size[2]),
            nn.Softmax(dim=1)
        )

        self.flatten = nn.Flatten(start_dim=0)

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        cnn_feat = self.cnn_1(x)
        pred_con = self.dnn_1(cnn_feat)
        return pred_con, argmax(pred_con, 1)


@models_available.register
class cnn2D_LSTM_v2_testphase(__model_settings_common):
    """Class of a 2D convolutional Decoding for feature extraction 06/2024"""
    def __init__(self, num_clusters=1, input_samples=12, output_samples=3):
        super().__init__('CNN+LSTM')
        self.model_shape = (1, num_clusters, 10, 10, input_samples)
        self.model_embedded = False
        self.input_samples = input_samples
        self.num_clusters = num_clusters
        self.output_samples = output_samples

        # --- Settings for DNN
        dense_layer_size = [64, 32]
        do_bias_train = True

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_clusters,
                out_channels=10,
                kernel_size=(3,3),
                stride= 1,
                padding=0
            ),
            nn.BatchNorm2d(10),
            nn.ReLU(),

        )
        self.dnn_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features= dense_layer_size[0],
                out_features= dense_layer_size[1],
                bias=do_bias_train
            ),
            nn.BatchNorm1d(dense_layer_size[1]),
            nn.ReLU(),
            nn.Linear(
                in_features= dense_layer_size[1],
                out_features= output_samples,
                bias=do_bias_train
            ),
            nn.BatchNorm1d(output_samples),
            nn.Softmax()
        )
        self.flatten = nn.Flatten()

        self.lstm = nn.LSTM(
            input_size=640,  # Adjust based on the output size from CNN
            hidden_size=64,          # Example hidden size, adjust as needed
            num_layers=input_samples,            # Example number of LSTM layers, adjust as needed
            batch_first=True         # Input and output tensors are provided as (batch, seq, feature)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        batch_size = x.size(0)
        x = x.view(batch_size*self.input_samples, self.num_clusters, 10, 10)
        cnn_feat = self.cnn_1(x)
        cnn_feat = cnn_feat.view(batch_size, self.input_samples, -1)
        lstm_output, _ = self.lstm(cnn_feat)
        pred_con = self.dnn_1(lstm_output[:,-1,:])

        return pred_con, argmax(pred_con, 1)
