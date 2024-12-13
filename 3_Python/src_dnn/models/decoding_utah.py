from torch import nn, Tensor, argmax, cat


class klaes_cnn_lstm_dec_v3(nn.Module):
    """Class of a convolutional Decoding for feature extraction"""
    def __init__(self, num_clusters, input_samples, output_samples=3):
        super().__init__()
        self.model_shape = (1, num_clusters, 10, 10, input_samples)
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

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        batch_size, num_clusters, height, width, num_time_windows = x.shape
        video = []
        for i in range(num_time_windows):
            img = x[:, :, :, :, i]
            img = img.view(batch_size, 2, 10, 10)
            cnn_out = self.cnn_1(img)
            cnn_out = cnn_out.view(batch_size, -1)
            video.append(cnn_out.unsqueeze(1))
        lstm_input = cat(video, dim=1)
        lstm_output, _ = self.lstm(lstm_input)
        pred_con = self.dnn_1(lstm_output[:, -1, :])
        return pred_con, argmax(pred_con, 1)


class klaes_cnn_lstm_dec_v2(nn.Module):
    """Class of a convolutional Decoding for feature extraction"""

    def __init__(self, num_clusters=1, input_samples=12, output_samples=3):
        super().__init__()
        self.model_shape = (1, num_clusters, 10, 10, input_samples)
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

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        batch_size, num_clusters, height, width, num_time_windows = x.shape
        video = []
        for i in range(num_time_windows):
            img = x[:, :, :, :, i]
            img.view(batch_size, num_clusters, 10, 10)
            cnn_out = self.cnn_1(img)
            cnn_out = cnn_out.view(batch_size, -1)
            video.append(cnn_out.unsqueeze(1))
        lstm_input = cat(video, dim=1)
        lstm_output, _ = self.lstm(lstm_input)
        pred_con = self.dnn_1(lstm_output[:, -1, :])
        return pred_con, argmax(pred_con, 1)


class klaes_cnn_lstm_dec_v1(nn.Module):
    """Class of a convolutional Decoding for feature extraction"""
    def __init__(self, num_clusters=1, input_samples=12, output_samples=3):
        super().__init__()
        self.model_shape = (1, num_clusters, 10, 10, input_samples)
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

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        batch_size, num_clusters, height, width, num_time_windows = x.shape
        video = []
        for i in range(num_time_windows):
            img = x[:, :, :, :, i]
            img.view(batch_size, num_clusters, 10, 10)
            cnn_out = self.cnn_1(img)
            cnn_out = cnn_out.view(batch_size, -1)
            video.append(cnn_out.unsqueeze(1))
        lstm_input = cat(video, dim=1)
        lstm_output, _ = self.lstm(lstm_input)
        pred_con = self.dnn_1(lstm_output[:,-1,:])
        return pred_con, argmax(pred_con, 1)
