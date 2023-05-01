import sys
import mne
import numpy as np
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
from multiprocessing import Process
import pickle
import argparse


def pretext_train_test(root_folder, k, N, epoch_sec):
    # get all data indices
    all_index = sorted(
        [int(path[6:12]) - 200000 for path in os.listdir(root_folder + "shhs1")]
    )
    sample_process(root_folder, k, N, epoch_sec, all_index)


def sample_process(root_folder, k, N, epoch_sec, index):
    # process each EEG sample: further split the samples into window sizes and using multiprocess
    for i, j in enumerate(index):
        if i % N == k:
            if k == 0:
                print("Progress: {} / {}".format(i, len(index)))

            # load the signal "X" part
            data = mne.io.read_raw_edf(
                root_folder + "shhs1/" + "shhs1-" + str(200000 + j) + ".edf"
            )
            X = data.get_data()

            # some EEG signals have missing channels, we treat them separately
            if X.shape[0] == 16:
                X = X[[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15], :]
            elif X.shape[0] == 15:
                X = X[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14], :]
            X = X[[2, 7], :]

            # slice the EEG signals into non-overlapping windows, window size = sampling rate per second * second time = 125 * windowsize
            for slice_index in range(X.shape[1] // (125 * epoch_sec)):
                path = (
                    root_folder
                    + "processed/shhs1-"
                    + str(200000 + j)
                    + "-"
                    + str(slice_index)
                    + ".pkl"
                )
                pickle.dump(
                    X[
                        :,
                        slice_index
                        * 125
                        * epoch_sec : (slice_index + 1)
                        * 125
                        * epoch_sec,
                    ],
                    open(path, "wb"),
                )


"""
SHHS dataset is downloaded from https://sleepdata.org/datasets/shhs
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windowsize", type=int, default=30, help="unit (seconds)")
    parser.add_argument(
        "--multiprocess", type=int, default=30, help="How many processes to use"
    )
    args = parser.parse_args()

    if not os.path.exists("./processed/"):
        os.makedirs("./processed/")

    root_folder = "/srv/local/data/SHHS/"

    N, epoch_sec = args.multiprocess, args.windowsize
    p_list = []
    for k in range(N):
        process = Process(
            target=pretext_train_test, args=(root_folder, k, N, epoch_sec)
        )
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()
