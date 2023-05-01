import pickle
import torch
import numpy as np
import torch.nn.functional as F
import os
from scipy.signal import resample
from scipy.signal import butter, iirnotch, filtfilt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter


class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # from default 200Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class CHBMITLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # 2560 -> 2000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class PTBLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=500):
        self.root = root
        self.files = files
        self.default_rate = 500
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, self.freq * 5, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        # 256 * 5 -> 1000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y


class HARLoader(torch.utils.data.Dataset):
    def __init__(self, dir, list_IDs, sampling_rate=50):
        self.list_IDs = list_IDs
        self.dir = dir
        self.label_map = ["1", "2", "3", "4", "5", "6"]
        self.default_rate = 50
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        sample = pickle.load(open(path, "rb"))
        X, y = sample["X"], self.label_map.index(sample["y"])
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(2.56 * self.sampling_rate), axis=-1)
        X = X / (
            np.quantile(
                np.abs(X), q=0.95, interpolation="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        return torch.FloatTensor(X), y


class UnsupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, root_prest, root_shhs):

        # prest dataset
        self.root_prest = root_prest
        exception_files = ["319431_data.npy"]
        self.prest_list = list(
            filter(
                lambda x: ("data" in x) and (x not in exception_files),
                os.listdir(self.root_prest),
            )
        )

        PREST_LENGTH = 2000
        WINDOW_SIZE = 200

        print("(prest) unlabeled data size:", len(self.prest_list) * 16)
        self.prest_idx_all = np.arange(PREST_LENGTH // WINDOW_SIZE)
        self.prest_mask_idx_N = PREST_LENGTH // WINDOW_SIZE // 3

        SHHS_LENGTH = 6000
        # shhs dataset
        self.root_shhs = root_shhs
        self.shhs_list = os.listdir(self.root_shhs)
        print("(shhs) unlabeled data size:", len(self.shhs_list))
        self.shhs_idx_all = np.arange(SHHS_LENGTH // WINDOW_SIZE)
        self.shhs_mask_idx_N = SHHS_LENGTH // WINDOW_SIZE // 5

    def __len__(self):
        return len(self.prest_list) + len(self.shhs_list)

    def prest_load(self, index):
        sample_path = self.prest_list[index]
        # (16, 16, 2000), 10s
        samples = np.load(os.path.join(self.root_prest, sample_path)).astype("float32")

        # find all zeros or all 500 signals and then remove them
        samples_max = np.max(samples, axis=(1, 2))
        samples_min = np.min(samples, axis=(1, 2))
        valid = np.where((samples_max > 0) & (samples_min < 0))[0]
        valid = np.random.choice(valid, min(8, len(valid)), replace=False)
        samples = samples[valid]

        # normalize samples (remove the amplitude)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 0

    def shhs_load(self, index):
        sample_path = self.shhs_list[index]
        # (2, 3750) sampled at 125
        sample = pickle.load(open(os.path.join(self.root_shhs, sample_path), "rb"))
        # (2, 6000) resample to 200
        samples = resample(sample, 6000, axis=-1)

        # normalize samples (remove the amplitude)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        # generate samples and targets and mask_indices
        samples = torch.FloatTensor(samples)

        return samples, 1

    def __getitem__(self, index):
        if index < len(self.prest_list):
            return self.prest_load(index)
        else:
            index = index - len(self.prest_list)
            return self.shhs_load(index)


def collate_fn_unsupervised_pretrain(batch):
    prest_samples, shhs_samples = [], []
    for sample, flag in batch:
        if flag == 0:
            prest_samples.append(sample)
        else:
            shhs_samples.append(sample)

    shhs_samples = torch.stack(shhs_samples, 0)
    if len(prest_samples) > 0:
        prest_samples = torch.cat(prest_samples, 0)
        return prest_samples, shhs_samples
    return 0, shhs_samples


class EEGSupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, tuev_data, chb_mit_data, iiic_data, tuab_data):
        # for TUEV
        tuev_root, tuev_files = tuev_data
        self.tuev_root = tuev_root
        self.tuev_files = tuev_files
        self.tuev_size = len(self.tuev_files)

        # for CHB-MIT
        chb_mit_root, chb_mit_files = chb_mit_data
        self.chb_mit_root = chb_mit_root
        self.chb_mit_files = chb_mit_files
        self.chb_mit_size = len(self.chb_mit_files)

        # for IIIC seizure
        iiic_x, iiic_y = iiic_data
        self.iiic_x = iiic_x
        self.iiic_y = iiic_y
        self.iiic_size = len(self.iiic_x)

        # for TUAB
        tuab_root, tuab_files = tuab_data
        self.tuab_root = tuab_root
        self.tuab_files = tuab_files
        self.tuab_size = len(self.tuab_files)

    def __len__(self):
        return self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size

    def tuev_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuev_root, self.tuev_files[index]), "rb")
        )
        X = sample["signal"]
        # 256 * 5 -> 1000
        X = resample(X, 1000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y, 0

    def chb_mit_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.chb_mit_root, self.chb_mit_files[index]), "rb")
        )
        X = sample["X"]
        # 2560 -> 2000
        X = resample(X, 2000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 1

    def iiic_load(self, index):
        data = self.iiic_x[index]
        samples = torch.FloatTensor(data)
        samples = samples / (
            torch.quantile(torch.abs(samples), q=0.95, dim=-1, keepdim=True) + 1e-8
        )
        y = np.argmax(self.iiic_y[index])
        return samples, y, 2

    def tuab_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuab_root, self.tuab_files[index]), "rb")
        )
        X = sample["X"]
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 3

    def __getitem__(self, index):
        if index < self.tuev_size:
            return self.tuev_load(index)
        elif index < self.tuev_size + self.chb_mit_size:
            index = index - self.tuev_size
            return self.chb_mit_load(index)
        elif index < self.tuev_size + self.chb_mit_size + self.iiic_size:
            index = index - self.tuev_size - self.chb_mit_size
            return self.iiic_load(index)
        elif (
            index < self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size
        ):
            index = index - self.tuev_size - self.chb_mit_size - self.iiic_size
            return self.tuab_load(index)
        else:
            raise ValueError("index out of range")


def collate_fn_supervised_pretrain(batch):
    tuev_samples, tuev_labels = [], []
    iiic_samples, iiic_labels = [], []
    chb_mit_samples, chb_mit_labels = [], []
    tuab_samples, tuab_labels = [], []

    for sample, labels, idx in batch:
        if idx == 0:
            tuev_samples.append(sample)
            tuev_labels.append(labels)
        elif idx == 1:
            iiic_samples.append(sample)
            iiic_labels.append(labels)
        elif idx == 2:
            chb_mit_samples.append(sample)
            chb_mit_labels.append(labels)
        elif idx == 3:
            tuab_samples.append(sample)
            tuab_labels.append(labels)
        else:
            raise ValueError("idx out of range")

    if len(tuev_samples) > 0:
        tuev_samples = torch.stack(tuev_samples)
        tuev_labels = torch.LongTensor(tuev_labels)
    if len(iiic_samples) > 0:
        iiic_samples = torch.stack(iiic_samples)
        iiic_labels = torch.LongTensor(iiic_labels)
    if len(chb_mit_samples) > 0:
        chb_mit_samples = torch.stack(chb_mit_samples)
        chb_mit_labels = torch.LongTensor(chb_mit_labels)
    if len(tuab_samples) > 0:
        tuab_samples = torch.stack(tuab_samples)
        tuab_labels = torch.LongTensor(tuab_labels)

    return (
        (tuev_samples, tuev_labels),
        (iiic_samples, iiic_labels),
        (chb_mit_samples, chb_mit_labels),
        (tuab_samples, tuab_labels),
    )


# define focal loss on binary classification
def focal_loss(y_hat, y, alpha=0.8, gamma=0.7):
    # y_hat: (N, 1)
    # y: (N, 1)
    # alpha: float
    # gamma: float
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    # y_hat = torch.clamp(y_hat, -75, 75)
    p = torch.sigmoid(y_hat)
    loss = -alpha * (1 - p) ** gamma * y * torch.log(p) - (1 - alpha) * p**gamma * (
        1 - y
    ) * torch.log(1 - p)
    return loss.mean()


# define binary cross entropy loss
def BCE(y_hat, y):
    # y_hat: (N, 1)
    # y: (N, 1)
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    loss = (
        -y * y_hat
        + torch.log(1 + torch.exp(-torch.abs(y_hat)))
        + torch.max(y_hat, torch.zeros_like(y_hat))
    )
    return loss.mean()
