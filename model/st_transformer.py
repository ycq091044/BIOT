import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


class PatchSTEmbedding(nn.Module):
    def __init__(self, emb_size, n_channels=16):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(n_channels, 64, 15, 8),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 256, 15, 8),
            Rearrange("b c s -> b s c"),
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, sequence_num=2000, inter=100, n_channels=16):
        super(ChannelAttention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(
            self.sequence_num / self.inter
        )  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.LayerNorm(
                n_channels
            ),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3),
        )
        self.key = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_channels),
            nn.Dropout(0.3),
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_channels),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, "b c s->b s c")
        temp_query = rearrange(self.query(temp), "b s c -> b c s")
        temp_key = rearrange(self.key(temp), "b s c -> b c s")

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = (
            torch.einsum("b c s, b m s -> b c m", channel_query, channel_key) / scaling
        )

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum("b c s, b c m -> b c s", x, channel_atten_score)
        """
        projections after or before multiplying with attention score are almost the same.
        """
        out = rearrange(out, "b c s -> b s c")
        out = self.projection(out)
        out = rearrange(out, "b s c -> b c s")
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum(
            "bhqd, bhkd -> bhqk", queries, keys
        )  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self, emb_size, num_heads=8, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class STTransformer(nn.Module):
    """
    Refer to https://arxiv.org/abs/2106.11170
    Modified from https://github.com/eeyhsong/EEG-Transformer
    """

    def __init__(
        self,
        emb_size=256,
        depth=3,
        n_classes=4,
        channel_legnth=2000,
        n_channels=16,
        **kwargs
    ):
        super().__init__()
        self.channel_attension = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(channel_legnth),
                ChannelAttention(n_channels=n_channels),
                nn.Dropout(0.5),
            )
        )
        self.patch_embedding = PatchSTEmbedding(emb_size, n_channels)
        self.transformer = TransformerEncoder(depth, emb_size)
        self.classification = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        x = self.channel_attension(x)
        x = self.patch_embedding(x)
        x = self.transformer(x).mean(dim=1)
        x = self.classification(x)
        return x


if __name__ == "__main__":
    X = torch.randn(2, 16, 2000)
    model = STTransformer(n_classes=6)
    out = model(X)
    print(out.shape)
