import os
import pickle

from multiprocessing import Pool
import numpy as np
import mne

# we need these channels
# (signals[signal_names['EEG FP1-REF']] - signals[signal_names['EEG F7-REF']],  # 0
# (signals[signal_names['EEG F7-REF']] - signals[signal_names['EEG T3-REF']]),  # 1
# (signals[signal_names['EEG T3-REF']] - signals[signal_names['EEG T5-REF']]),  # 2
# (signals[signal_names['EEG T5-REF']] - signals[signal_names['EEG O1-REF']]),  # 3
# (signals[signal_names['EEG FP2-REF']] - signals[signal_names['EEG F8-REF']]),  # 4
# (signals[signal_names['EEG F8-REF']] - signals[signal_names['EEG T4-REF']]),  # 5
# (signals[signal_names['EEG T4-REF']] - signals[signal_names['EEG T6-REF']]),  # 6
# (signals[signal_names['EEG T6-REF']] - signals[signal_names['EEG O2-REF']]),  # 7
# (signals[signal_names['EEG FP1-REF']] - signals[signal_names['EEG F3-REF']]),  # 14
# (signals[signal_names['EEG F3-REF']] - signals[signal_names['EEG C3-REF']]),  # 15
# (signals[signal_names['EEG C3-REF']] - signals[signal_names['EEG P3-REF']]),  # 16
# (signals[signal_names['EEG P3-REF']] - signals[signal_names['EEG O1-REF']]),  # 17
# (signals[signal_names['EEG FP2-REF']] - signals[signal_names['EEG F4-REF']]),  # 18
# (signals[signal_names['EEG F4-REF']] - signals[signal_names['EEG C4-REF']]),  # 19
# (signals[signal_names['EEG C4-REF']] - signals[signal_names['EEG P4-REF']]),  # 20
# (signals[signal_names['EEG P4-REF']] - signals[signal_names['EEG O2-REF']]))) # 21
standard_channels = [
    "EEG FP1-REF",
    "EEG F7-REF",
    "EEG T3-REF",
    "EEG T5-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F8-REF",
    "EEG T4-REF",
    "EEG T6-REF",
    "EEG O2-REF",
    "EEG FP1-REF",
    "EEG F3-REF",
    "EEG C3-REF",
    "EEG P3-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F4-REF",
    "EEG C4-REF",
    "EEG P4-REF",
    "EEG O2-REF",
]


def split_and_dump(params):
    fetch_folder, sub, dump_folder, label = params
    for file in os.listdir(fetch_folder):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            raw.resample(200)
            ch_name = raw.ch_names
            raw_data = raw.get_data()
            channeled_data = raw_data.copy()[:16]
            try:
                channeled_data[0] = (
                    raw_data[ch_name.index("EEG FP1-REF")]
                    - raw_data[ch_name.index("EEG F7-REF")]
                )
                channeled_data[1] = (
                    raw_data[ch_name.index("EEG F7-REF")]
                    - raw_data[ch_name.index("EEG T3-REF")]
                )
                channeled_data[2] = (
                    raw_data[ch_name.index("EEG T3-REF")]
                    - raw_data[ch_name.index("EEG T5-REF")]
                )
                channeled_data[3] = (
                    raw_data[ch_name.index("EEG T5-REF")]
                    - raw_data[ch_name.index("EEG O1-REF")]
                )
                channeled_data[4] = (
                    raw_data[ch_name.index("EEG FP2-REF")]
                    - raw_data[ch_name.index("EEG F8-REF")]
                )
                channeled_data[5] = (
                    raw_data[ch_name.index("EEG F8-REF")]
                    - raw_data[ch_name.index("EEG T4-REF")]
                )
                channeled_data[6] = (
                    raw_data[ch_name.index("EEG T4-REF")]
                    - raw_data[ch_name.index("EEG T6-REF")]
                )
                channeled_data[7] = (
                    raw_data[ch_name.index("EEG T6-REF")]
                    - raw_data[ch_name.index("EEG O2-REF")]
                )
                channeled_data[8] = (
                    raw_data[ch_name.index("EEG FP1-REF")]
                    - raw_data[ch_name.index("EEG F3-REF")]
                )
                channeled_data[9] = (
                    raw_data[ch_name.index("EEG F3-REF")]
                    - raw_data[ch_name.index("EEG C3-REF")]
                )
                channeled_data[10] = (
                    raw_data[ch_name.index("EEG C3-REF")]
                    - raw_data[ch_name.index("EEG P3-REF")]
                )
                channeled_data[11] = (
                    raw_data[ch_name.index("EEG P3-REF")]
                    - raw_data[ch_name.index("EEG O1-REF")]
                )
                channeled_data[12] = (
                    raw_data[ch_name.index("EEG FP2-REF")]
                    - raw_data[ch_name.index("EEG F4-REF")]
                )
                channeled_data[13] = (
                    raw_data[ch_name.index("EEG F4-REF")]
                    - raw_data[ch_name.index("EEG C4-REF")]
                )
                channeled_data[14] = (
                    raw_data[ch_name.index("EEG C4-REF")]
                    - raw_data[ch_name.index("EEG P4-REF")]
                )
                channeled_data[15] = (
                    raw_data[ch_name.index("EEG P4-REF")]
                    - raw_data[ch_name.index("EEG O2-REF")]
                )
            except:
                with open("tuab-process-error-files.txt", "a") as f:
                    f.write(file + "\n")
                continue
            for i in range(channeled_data.shape[1] // 2000):
                dump_path = os.path.join(
                    dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
                )
                pickle.dump(
                    {"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": label},
                    open(dump_path, "wb"),
                )


if __name__ == "__main__":
    """
    TUAB dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    """
    # root to abnormal dataset
    root = "/srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf/"
    channel_std = "01_tcp_ar"

    # train, val abnormal subjects
    train_val_abnormal = os.path.join(root, "train", "abnormal", channel_std)
    train_val_a_sub = list(
        set([item.split("_")[0] for item in os.listdir(train_val_abnormal)])
    )
    np.random.shuffle(train_val_a_sub)
    train_a_sub, val_a_sub = (
        train_val_a_sub[: int(len(train_val_a_sub) * 0.8)],
        train_val_a_sub[int(len(train_val_a_sub) * 0.8) :],
    )

    # train, val normal subjects
    train_val_normal = os.path.join(root, "train", "normal", channel_std)
    train_val_n_sub = list(
        set([item.split("_")[0] for item in os.listdir(train_val_normal)])
    )
    np.random.shuffle(train_val_n_sub)
    train_n_sub, val_n_sub = (
        train_val_n_sub[: int(len(train_val_n_sub) * 0.8)],
        train_val_n_sub[int(len(train_val_n_sub) * 0.8) :],
    )

    # test abnormal subjects
    test_abnormal = os.path.join(root, "eval", "abnormal", channel_std)
    test_a_sub = list(set([item.split("_")[0] for item in os.listdir(test_abnormal)]))

    # test normal subjects
    test_normal = os.path.join(root, "eval", "normal", channel_std)
    test_n_sub = list(set([item.split("_")[0] for item in os.listdir(test_normal)]))

    # create the train, val, test sample folder
    if not os.path.exists(os.path.join(root, "processed")):
        os.makedirs(os.path.join(root, "processed"))

    if not os.path.exists(os.path.join(root, "processed", "train")):
        os.makedirs(os.path.join(root, "processed", "train"))
    train_dump_folder = os.path.join(root, "processed", "train")

    if not os.path.exists(os.path.join(root, "processed", "val")):
        os.makedirs(os.path.join(root, "processed", "val"))
    val_dump_folder = os.path.join(root, "processed", "val")

    if not os.path.exists(os.path.join(root, "processed", "test")):
        os.makedirs(os.path.join(root, "processed", "test"))
    test_dump_folder = os.path.join(root, "processed", "test")

    # fetch_folder, sub, dump_folder, labels
    parameters = []
    for train_sub in train_a_sub:
        parameters.append([train_val_abnormal, train_sub, train_dump_folder, 1])
    for train_sub in train_n_sub:
        parameters.append([train_val_normal, train_sub, train_dump_folder, 0])
    for val_sub in val_a_sub:
        parameters.append([train_val_abnormal, val_sub, val_dump_folder, 1])
    for val_sub in val_n_sub:
        parameters.append([train_val_normal, val_sub, val_dump_folder, 0])
    for test_sub in test_a_sub:
        parameters.append([test_abnormal, test_sub, test_dump_folder, 1])
    for test_sub in test_n_sub:
        parameters.append([test_normal, test_sub, test_dump_folder, 0])

    # split and dump in parallel
    with Pool(processes=24) as pool:
        # Use the pool.map function to apply the square function to each element in the numbers list
        result = pool.map(split_and_dump, parameters)
