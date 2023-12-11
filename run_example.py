import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl

from model import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,
    FFCL,
    STTransformer,
    BIOTClassifier,
    UnsupervisedPretrain,
)

# Sample data (batch_size, n_channels, sample_length)
x = torch.randn(64, 16, 2000)

"""
SPaRCNet - 1D CNN DenseNet
ContraWR - Spectrogram + 2D CNN
CNNTransformer - Split into windows + 2D CNN + Transformer
FFCL - CNN + LSTM combined encoder
STTransformer - multilevel Transformer (channel-wise and temporal)
BIOT - biosignal tokenization + Linformer
BIOT-pretrain-PREST - pre-trained BIOT model on 5M EEG data
BIOT-pretrain-SHHS+PREST - pre-trained BIOT model on 5M+5M EEG data
BIOT-pretrain-dix-datasets - pre-trained BIOT model on all six EEG data
"""

model_name = sys.argv[1]

if model_name == "SPaRCNet":
    sparcnet = SPaRCNet(
        in_channels=16,
        sample_length=2000,
        n_classes=5,
        block_layers=4,
        growth_rate=16,
        bn_size=16,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
    )
    out = sparcnet(x)
    print(out.shape)


elif model_name == "ContraWR":
    contrawr = ContraWR(in_channels=16, n_classes=5, fft=200, steps=20)
    out = contrawr(x)
    print(out.shape)


elif model_name == "CNNTransformer":
    cnn_transformer = CNNTransformer(
        in_channels=16,
        n_classes=5,
        fft=200,
        steps=20,
        dropout=0.2,
        nhead=4,
        emb_size=256,
    )
    out = cnn_transformer(x)
    print(out.shape)


elif model_name == "FFCL":
    ffcl = FFCL(
        in_channels=16,
        n_classes=5,
        fft=200,
        steps=20,
        sample_length=2000,
        shrink_steps=20,
    )
    out = ffcl(x)
    print(out.shape)


elif model_name == "STTransformer":
    st_transformer = STTransformer(emb_size=256, depth=4, n_classes=5)
    out = st_transformer(x)
    print(out.shape)


elif model_name == "BIOT":
    biot_classifier = BIOTClassifier(
        emb_size=256, heads=8, depth=4, n_classes=5, n_fft=200, hop_length=100
    )
    out = biot_classifier(x)
    print(out.shape)


elif model_name == "BIOT-pretrain-PREST":
    pretrained_model_path = "pretrained-models/EEG-PREST-16-channels.ckpt"
    biot_classifier = BIOTClassifier(
        emb_size=256,
        heads=8,
        depth=4,
        n_classes=5,
        n_fft=200,
        hop_length=100,
        n_channels=16,  # here is 16
    )
    biot_classifier.biot.load_state_dict(torch.load(pretrained_model_path))
    out = biot_classifier(x)
    print(out.shape)


elif model_name == "BIOT-pretrain-SHHS+PREST":
    pretrained_model_path = "pretrained-models/EEG-SHHS+PREST-18-channels.ckpt"
    biot_classifier = BIOTClassifier(
        emb_size=256,
        heads=8,
        depth=4,
        n_classes=5,
        n_fft=200,
        hop_length=100,
        n_channels=18,  # here is 18
    )
    biot_classifier.biot.load_state_dict(torch.load(pretrained_model_path))
    out = biot_classifier(x)
    print(out.shape)


elif model_name == "BIOT-pretrain-six-datasets":
    pretrained_model_path = "pretrained-models/EEG-six-datesets-18-channels.ckpt"

    biot_classifier = BIOTClassifier(
        emb_size=256,
        heads=8,
        depth=4,
        n_classes=5,
        n_fft=200,
        hop_length=100,
        n_channels=18,  # here is 18
    )
    biot_classifier.biot.load_state_dict(torch.load(pretrained_model_path))
    out = biot_classifier(x)
    print(out.shape)


elif model_name == "BIOT-unsupervised":
    biot_unsupervised = UnsupervisedPretrain(
        emb_size=256, heads=8, depth=4, n_channels=18, n_fft=200, hop_length=100
    )
    out1, out2 = biot_unsupervised(x)
    print(out1.shape, out2.shape)


else:
    raise NotImplementedError
