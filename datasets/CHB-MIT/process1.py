import os
from collections import defaultdict
import pyedflib
import pyedflib.highlevel as hl
import numpy as np
import copy
import shutil
import bz2
import pickle
import _pickle as cPickle
import multiprocessing as mp


# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
    # with bz2.BZ2File(title + '.pbz2', 'w') as f:
    #     cPickle.dump(data, f)
    pickle.dump(data, open(title, "wb"))


# Process metadata
def process_metadata(summary, filename):
    f = open(summary, "r")

    metadata = {}
    lines = f.readlines()
    times = []
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) == 3 and line[2] == filename:
            j = i + 1
            processed = False
            while not processed:
                if lines[j].split()[0] == "Number":
                    seizures = int(lines[j].split()[-1])
                    processed = True
                j = j + 1

            # If file has seizures get start and end time
            if seizures > 0:
                j = i + 1
                for s in range(seizures):
                    # Save start and end time of each seizure
                    processed = False
                    while not processed:
                        l = lines[j].split()
                        # print(l)

                        if l[0] == "Seizure" and "Start" in l:
                            start = int(l[-2]) * 256 - 1  # Index of start time
                            end = (
                                int(lines[j + 1].split()[-2]) * 256 - 1
                            )  # Index of end time
                            processed = True
                        j = j + 1
                    times.append((start, end))

            metadata["seizures"] = seizures
            metadata["times"] = times

    return metadata


# Keep some channels from a .edf and ignore the others
def drop_channels(edf_source, edf_target=None, to_keep=None, to_drop=None):
    signals, signal_headers, header = hl.read_edf(
        edf_source, ch_nrs=to_keep, digital=False
    )
    clean_file = {}
    for signal, header in zip(signals, signal_headers):
        channel = header.get("label")
        if channel in clean_file.keys():
            channel = channel + "-2"
        clean_file[channel] = signal
    return clean_file


# At first, it permuted the channels of a edf signal
# Now, only keeps valid channels and compress+save into pkl
def move_channels(clean_dict, channels, target):
    # Keep only valid channels
    keys_to_delete = []
    for key in clean_dict:
        if key != "metadata" and key not in channels.keys():
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del clean_dict[key]

    # Get size of the numpy array
    size = 0
    for item in clean_dict.keys():
        if item != "metadata":
            size = len(clean_dict.get(item))
            break

    for k in channels.keys():
        if k not in clean_dict.keys():
            clean_dict[k] = np.zeros(size, dtype=float)

    compressed_pickle(target + ".pkl", clean_dict)


# Process edf files of a pacient from start number to end number
def process_files(pacient, valid_channels, channels, start, end):
    for num in range(start, end + 1):
        to_keep = []

        num = ("0" + str(num))[-2:]
        filename = "{path}/chb{p}/chb{p}_{n}.edf".format(
            path=signals_path, p=pacient, n=num
        )

        # Check with (cleaned) reference file  if we have to remove more channels
        try:
            signals, signal_headers, header = hl.read_edf(filename, digital=False)
            n = 0
            for h in signal_headers:
                if h.get("label") in valid_channels:
                    if n not in to_keep:
                        to_keep.append(n)
                n = n + 1

        except OSError:
            print("****************************************")
            print("WARNING - Do not worry")
            print("File", filename, "does not exist.\nProcessing next file.")
            print("****************************************")
            continue

        if len(to_keep) > 0:
            try:
                print(
                    "Removing",
                    len(signal_headers) - len(to_keep),
                    "channels from file ",
                    "chb{p}_{n}.edf".format(p=pacient, n=num),
                )
                clean_dict = drop_channels(
                    filename,
                    edf_target="{path}/chb{p}/chb{p}_{n}.edf".format(
                        path=clean_path, p=pacient, n=num
                    ),
                    to_keep=to_keep,
                )
                print("Processing file ", filename)
            except AssertionError:
                print("****************************************")
                print("WARNING - Do not worry")
                print("File", filename, "does not exist.\nProcessing next file.")
                print("****************************************")
                continue

        metadata = process_metadata(
            "{path}/chb{p}/chb{p}-summary.txt".format(path=signals_path, p=pacient),
            "chb{p}_{n}.edf".format(p=pacient, n=num),
        )
        metadata["channels"] = valid_channels
        clean_dict["metadata"] = metadata
        target = "{path}/chb{p}/chb{p}_{n}.edf".format(
            path=clean_path, p=pacient, n=num
        )
        move_channels(clean_dict, channels, target)


def start_process(pacient, num, start, end, sum_ind):
    # Summary file
    f = open(
        "{path}/chb{p}/chb{p}-summary.txt".format(path=signals_path, p=pacient), "r"
    )

    channels = defaultdict(list)  # Dict of channels and indices
    valid_channels = []  # Valid channels
    to_keep = []  # Indices of channels we want to keep

    channel_index = 1  # Index for each channel
    summary_index = 0  # Index to choose which channel reference take from summary file

    # Process summary file
    for line in f:
        line = line.split()
        if len(line) == 0:
            continue

        if line[0] == "Channels" and line[1] == "changed:":
            summary_index += 1

        if (
            line[0] == "Channel"
            and summary_index == sum_ind
            and (line[2] != "-" and line[2] != ".")
        ):  # '-' means a void channel
            if (
                line[2] in channels.keys()
            ):  # In case of repeated channel just add '-2' to the label
                name = line[2] + "-2"
            else:
                name = line[2]

            # Add channel to dict and update lists
            channels[name].append(str(channel_index))
            channel_index += 1
            valid_channels.append(name)
            to_keep.append(int(line[1][:-1]) - 1)

    # for item in channels.items(): print(item)

    # Clean reference file
    filename = "{path}/chb{p}/chb{p}_{n}.edf".format(
        path=signals_path, p=pacient, n=num
    )
    target = "{path}/chb{p}/chb{p}_{n}.edf".format(path=clean_path, p=pacient, n=num)

    if not os.path.exists("{path}/chb{p}".format(p=pacient, path=clean_path)):
        os.makedirs("{path}/chb{p}".format(p=pacient, path=clean_path))

    clean_dict = drop_channels(filename, edf_target=target, to_keep=to_keep)

    # Process metadata : Number of seizures and start/end time
    metadata = process_metadata(
        "{path}/chb{p}/chb{p}-summary.txt".format(path=signals_path, p=pacient),
        "chb{p}_{n}.edf".format(p=pacient, n=num),
    )

    metadata["channels"] = valid_channels
    clean_dict["metadata"] = metadata

    compressed_pickle(target + ".pkl", clean_dict)

    # Process the rest of the files to get same channels as reference file
    process_files(pacient, valid_channels, channels, start, end)


# PARAMETERS
signals_path = "/srv/local/data/physionet.org/files/chbmit/1.0.0"  # Path to the data main directory
clean_path = "/srv/local/data/physionet.org/files/chbmit/1.0.0/clean_signals"  # Path where to store clean data

if not os.path.exists(clean_path):
    os.makedirs(clean_path)

# Clean pacients one by one manually with these parameters
pacient = "04"
num = "01"  # Reference file
summary_index = 0  # Index of channels summary reference
start = 28  # Number of first file to process
end = 28  # Number of last file to process
# Start the process
# start_process(pacient, num, start, end, summary_index)


# FULL DATA PROCESS
parameters = [
    ("01", "01", 2, 46, 0),
    ("02", "01", 2, 35, 0),
    ("03", "01", 2, 38, 0),
    ("05", "01", 2, 39, 0),
    ("06", "01", 2, 24, 0),
    ("07", "01", 2, 19, 0),
    ("08", "02", 3, 29, 0),
    ("10", "01", 2, 89, 0),
    ("11", "01", 2, 99, 0),
    ("14", "01", 2, 42, 0),
    ("20", "01", 2, 68, 0),
    ("21", "01", 2, 33, 0),
    ("22", "01", 2, 77, 0),
    ("23", "06", 7, 20, 0),
    ("24", "01", 3, 21, 0),
    ("04", "07", 1, 43, 1),
    ("09", "02", 1, 19, 1),
    ("15", "02", 1, 63, 1),
    ("16", "01", 2, 19, 0),
    ("18", "02", 1, 36, 1),
    ("19", "02", 1, 30, 1),
]

with mp.Pool(mp.cpu_count()) as pool:
    res = pool.starmap(start_process, parameters)
