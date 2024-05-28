from torch import nn, Tensor, argmax, flatten
from package.dnn.pytorch_handler import __model_settings_common


class mnist_gki_v1(__model_settings_common):
    def __init__(self):
        super().__init__('Classifier')
        self.model = nn.Sequential(
            nn.Linear(784, 10),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_gki_v2(__model_settings_common):
    def __init__(self):
        super().__init__('Classifier')
        self.model = nn.Sequential(
            nn.Linear(784, 10),
            nn.BatchNorm1d(10),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_gki_v3(__model_settings_common):
    def __init__(self):
        super().__init__('Classifier')
        self.model = nn.Sequential(
            nn.Linear(784, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_gki_v4(__model_settings_common):
    def __init__(self):
        super().__init__('Classifier')
        self.model = nn.Sequential(
            nn.Linear(784, 10),
            nn.BatchNorm1d(10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_gki_v5(__model_settings_common):
    def __init__(self):
        super().__init__('Classifier')
        self.model = nn.Sequential(
            nn.Linear(784, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, 90),
            nn.BatchNorm1d(90),
            nn.ReLU(),
            nn.Linear(90, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_gki_v6(__model_settings_common):
    def __init__(self):
        super().__init__('Classifier')
        feature_size = [16, 24, 32]
        self.model_cnn = nn.Sequential(
            nn.Conv2d(1, feature_size[0], 4, 1, 1),
            nn.BatchNorm2d(num_features=feature_size[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(feature_size[0], feature_size[1], 4, 1, 1),
            nn.BatchNorm2d(num_features=feature_size[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(feature_size[1], feature_size[2], 4, 1, 1),
            nn.BatchNorm2d(num_features=feature_size[2]),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )
        self.model_cl = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Linear(24, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x0 = x.unsqueeze(dim=1)
        x0 = self.model_cnn(x0)
        x1 = flatten(x0, start_dim=2)
        x1 = flatten(x1, start_dim=1)
        prob = self.model_cl(x1)
        return prob, argmax(prob, 1)
