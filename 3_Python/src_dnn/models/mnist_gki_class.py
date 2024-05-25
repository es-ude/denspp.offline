from torch import nn, Tensor, argmax, flatten
from package.dnn.pytorch_handler import __model_settings_common


class mnist_gki_v1(__model_settings_common):
    def __init__(self):
        super().__init__('Classifier')
        self.model = nn.Sequential(
            nn.Linear(784, 10)
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
            nn.BatchNorm1d(10)
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
            nn.ReLU()
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
            nn.Linear(784, 10),
            nn.BatchNorm1d(10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_gki_v6(__model_settings_common):
    def __init__(self):
        super().__init__('Classifier')
        self.model = nn.Sequential(
            nn.Linear(784, 400),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.BatchNorm1d(10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)
