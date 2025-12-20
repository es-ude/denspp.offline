from torch import nn, Tensor, argmax, flatten, reshape


class mnist_test_cl_v0(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_shape = (1, 28, 28)

        self.model = nn.Sequential(
            nn.Linear(784, 10)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_test_cl_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_shape = (1, 28, 28)

        self.model = nn.Sequential(
            nn.Linear(784, 10),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_test_cl_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_shape = (1, 28, 28)

        self.model = nn.Sequential(
            nn.Linear(784, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_test_cl_v3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_shape = (1, 28, 28)

        self.model = nn.Sequential(
            nn.Linear(784, 10),
            nn.BatchNorm1d(10),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_test_cl_v4(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_shape = (1, 28, 28)

        self.model = nn.Sequential(
            nn.Linear(784, 10),
            nn.BatchNorm1d(10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_test_cl_v5(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_shape = (1, 28, 28)

        self.model = nn.Sequential(
            nn.Linear(784, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, 10),
            nn.BatchNorm1d(10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_test_cl_v6(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_shape = (1, 28, 28)

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
            nn.BatchNorm1d(10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class mnist_test_cl_v7(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_shape = (1, 28, 28)
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
            nn.BatchNorm1d(10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x0 = x.unsqueeze(dim=1)
        x0 = self.model_cnn(x0)
        x1 = flatten(x0, start_dim=2)
        x1 = flatten(x1, start_dim=1)
        prob = self.model_cl(x1)
        return prob, argmax(prob, 1)
