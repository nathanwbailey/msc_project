import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bidirectional=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bidirectional=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=input_size)

    def forward(self, input, hidden, cell):
        out, (hidden, cell) = self.lstm(input, (hidden, cell))
        prediction = self.fc(out.squeeze(0))
        return prediction, hidden, cell


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_len,
        bidirectional=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(
            input_size, hidden_size, num_layers, bidirectional
        )
        self.decoder = Decoder(
            input_size, hidden_size, num_layers, bidirectional
        )
        self.output_len = output_len

    def forward(self, source, target=None):
        hidden, cell = self.encoder(source)
        device = source.device
        decoder_input = source[:, -1, :].unsqueeze(1).to(device)
        outputs = []
        for t in range(self.output_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            if output.ndim == 2:
                output = output.unsqueeze(0)
            outputs.append(output)
            if target is not None:
                decoder_input = target[:, t, :].unsqueeze(1)
            else:
                decoder_input = output
        outputs = torch.cat(outputs, dim=1)
        return outputs
