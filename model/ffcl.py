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


class FFCL(nn.Module):
    """The whole model is CNN + LSTM. We combine the embeddings and add an FC layer."""

    def __init__(
        self,
        in_channels=16,
        n_classes=6,
        fft=200,
        steps=20,
        sample_length=2000,
        shrink_steps=20,
    ):
        super(FFCL, self).__init__()
        self.fft = fft
        self.steps = steps
        self.conv1 = ResBlock(in_channels, 32, 2, True, True)
        self.conv2 = ResBlock(32, 64, 2, True, True)
        self.conv3 = ResBlock(64, 128, 2, True, True)
        self.conv4 = ResBlock(128, 256, 2, True, True)

        self.lstm = nn.LSTM(
            input_size=sample_length // shrink_steps,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )
        self.shrink_steps = shrink_steps

        self.classifier = nn.Sequential(
            nn.ELU(),
            nn.Linear(256 * 2, n_classes),
        )

    def shorten(self, x):
        """
        x: (batch_size, n_channels, length)
        out: (batch_size, n_channels * shrink_steps, length // shrink_steps)
        """
        segments = [
            x[:, :, i :: self.shrink_steps] for i in range(0, self.shrink_steps)
        ]
        return torch.cat(segments, dim=1)

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

    def forward(self, x):
        e1 = self.torch_stft(x)
        e1 = self.conv1(e1)
        e1 = self.conv2(e1)
        e1 = self.conv3(e1)
        e1 = self.conv4(e1).squeeze(-1).squeeze(-1)

        e2 = self.shorten(x)
        e2 = self.lstm(e2)[0][:, -1]

        e = torch.cat([e1, e2], dim=1)
        return self.classifier(e)


if __name__ == "__main__":
    x = torch.randn(2, 16, 2000)
    model = FFCL(
        in_channels=16,
        n_classes=6,
        fft=200,
        steps=20,
        sample_length=2000,
        shrink_steps=20,
    )
    out = model(x)
    print(out.shape)
