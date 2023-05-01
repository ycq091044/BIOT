import math
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Convolutional Residual Block 2D
    This block stacks two convolutional layers with batch normalization,
    max pooling, dropout, and residual connection.
    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        stride: stride of the convolutional layers.
        downsample: whether to use a downsampling residual connection.
        pooling: whether to use max pooling.
    Example:
        >>> import torch
        >>> from pyhealth.models import ResBlock2D
        >>>
        >>> model = ResBlock2D(6, 16, 1, True, True)
        >>> input_ = torch.randn((16, 6, 28, 150))  # (batch, channel, height, width)
        >>> output = model(input_)
        >>> output.shape
        torch.Size([16, 16, 14, 75])
    """

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=False, pooling=False
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(3, stride=stride, padding=1)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CNNTransformer(nn.Module):
    def __init__(
        self,
        in_channels=16,
        n_classes=6,
        fft=200,
        steps=20,
        dropout=0.2,
        nhead=4,
        emb_size=256,
        n_segments=5,
    ):
        super().__init__()
        self.fft = fft
        self.steps = steps
        self.n_segments = n_segments
        self.conv1 = ResBlock(in_channels, 32, 2, True, True)
        self.conv2 = ResBlock(32, 64, 2, True, True)
        self.conv3 = ResBlock(64, 128, 2, True, True)
        self.conv4 = ResBlock(128, 256, 2, True, True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=nhead,
            dim_feedforward=emb_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.positional_encoding = PositionalEncoding(emb_size)

        self.classifier = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def torch_stft(self, x):
        signal = []
        for s in range(x.shape[1]):
            spectral = torch.stft(
                x[:, s, :],
                n_fft=self.fft,
                hop_length=self.fft // self.steps,
                win_length=self.fft,
                normalized=True,
                center=True,
                onesided=True,
                return_complex=True,
            )
            signal.append(spectral)
        stacked = torch.stack(signal).permute(1, 0, 2, 3)
        return torch.abs(stacked)

    def cnn(self, x):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x).squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        n_length = x.shape[2] // self.n_segments
        cnn_emb = [
            self.cnn(x[:, :, idx * n_length : idx * n_length + n_length]).unsqueeze(1)
            for idx in range(self.n_segments)
        ]
        # (batch, ts_steps, emb)
        x = torch.cat(cnn_emb, dim=1)
        # (batch, ts_steps, emb)
        x = self.positional_encoding(x)
        x = self.transformer(x).mean(dim=1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 16, 2000)
    model = CNNTransformer(in_channels=16, n_classes=6, fft=200, steps=2)
    out = model(x)
    print(out.shape)
