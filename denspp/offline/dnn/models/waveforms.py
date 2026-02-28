from torch import nn, Tensor, argmax, cat, flatten


class waveforms_mlp_cl_v0(nn.Module):
    def __init__(self, input_size: int=280, output_size: int=4):
        super().__init__()
        self.model_shape = (1, input_size)
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        config_network = [input_size, 40, output_size]

        # --- Model Deployment
        self.model = nn.Sequential()
        for idx, layer_size in enumerate(config_network[1:], start=1):
            self.model.add_module(f"linear_{idx:02d}", nn.Linear(in_features=config_network[idx-1], out_features=layer_size, bias=do_train_bias))
            self.model.add_module(f"batch1d_{idx:02d}", nn.BatchNorm1d(num_features=layer_size, affine=do_train_batch))
            if not idx == len(config_network)-1:
                self.model.add_module(f"act_{idx:02d}", nn.ReLU())
            else:
                # self.model.add_module(f"soft", nn.Softmax(dim=1))
                pass

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, dim=1)


class waveforms_mlp_ae_v0(nn.Module):
    def __init__(self, input_size: int=280, output_size: int=4):
        super().__init__()
        self.model_shape = (1, input_size)
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        config_network = [input_size, 120, 36, output_size]

        # --- Model Deployment: Encoder
        self.encoder = nn.Sequential()
        for idx, layer_size in enumerate(config_network[1:], start=1):
            self.encoder.add_module(f"linear_{idx:02d}", nn.Linear(in_features=config_network[idx - 1], out_features=layer_size, bias=do_train_bias))
            self.encoder.add_module(f"batch1d_{idx:02d}", nn.BatchNorm1d(num_features=layer_size, affine=do_train_batch))
            if not idx == len(config_network) - 1:
                self.encoder.add_module(f"act_{idx:02d}", nn.ReLU())

        # --- Model Deployment: Decoder
        self.decoder = nn.Sequential()
        for idx, layer_size in enumerate(reversed(config_network[:-1]), start=1):
            if idx == 1:
                self.decoder.add_module(f"act_dec_{idx:02d}", nn.ReLU())
            self.decoder.add_module(f"linear_{idx:02d}", nn.Linear(in_features=config_network[-idx], out_features=layer_size, bias=do_train_bias))
            if not idx == len(config_network) - 1:
                self.decoder.add_module(f"batch1d_{idx:02d}", nn.BatchNorm1d(num_features=layer_size, affine=do_train_batch))
                self.decoder.add_module(f"act_{idx:02d}", nn.ReLU())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class waveforms_lstm_cl_v0(nn.Module):
    def __init__(self, input_size: int=40, output_size: int=4):
        super().__init__()
        hidden_size = input_size

        self.lstm = nn.Sequential(
            nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Build a sequence - # (batch, seq_len, 1)
        if x.dim() == 2 and x.shape[1] != 1:
            x = x.unsqueeze(-1)

        lstm_out, (h_n, c_n) = self.lstm(x)
        if self.lstm[0].bidirectional:
            last_hidden = cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]

        logits = self.classifier(last_hidden)
        return logits, argmax(logits, dim=1)


class sinusoidal_lstm_cl_v0(nn.Module):
    def __init__(self, input_size: int=32, output_size: int=2):
        super().__init__()
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=1, hidden_size=input_size, batch_first=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(input_size, output_size)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: Build shape of (batch, seq_len, input_size)
        if x.dim() == 2 and x.shape[1] != 1:
            x = x.unsqueeze(-1)

        lstm_out, (h_n, c_n) = self.lstm(x)
        if self.lstm[0].bidirectional:
            last_hidden = cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits, argmax(logits, dim=1)
