import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = 10000 ** (torch.arange(0, emb_size, 2).float() / emb_size)
        pe = torch.zeros(max_len, emb_size)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        nhead=1,
        dropout=0,
        num_layers=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(
            x, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        nhead=1,
        dropout=0,
        num_layers=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        x,
        hidden_state,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
    ):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.decoder(
            x,
            hidden_state,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
        )
        return x


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        nhead=1,
        dropout=0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(
            input_size,
            hidden_size,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
        )
        self.decoder = Decoder(
            input_size,
            hidden_size,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
        )
        self.fc_layer = nn.Linear(hidden_size, input_size)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_padding_mask=None,
        tgt_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
    ):
        hidden_state = self.encoder(src, src_mask, src_padding_mask)
        output = self.decoder(
            tgt,
            hidden_state,
            tgt_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
        )
        output = self.fc_layer(output)
        return output
