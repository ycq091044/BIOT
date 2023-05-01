import pickle
import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

root = "/srv/local/data/physionet.org/files/chbmit/1.0.0/clean_signals"
out = "/srv/local/data/physionet.org/files/chbmit/1.0.0/clean_segments"

# root = 'clean_signals'
# out = 'clean_segments'

if not os.path.exists(out):
    os.makedirs(out)

# dump chb23 and chb24 to test, ch21 and ch22 to val, and the rest to train
test_pats = ["chb23", "chb24"]
val_pats = ["chb21", "chb22"]
train_pats = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
    "chb06",
    "chb07",
    "chb08",
    "chb09",
    "chb10",
    "chb11",
    "chb12",
    "chb13",
    "chb14",
    "chb15",
    "chb16",
    "chb17",
    "chb18",
    "chb19",
    "chb20",
]
channels = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]
SAMPLING_RATE = 256


def sub_to_segments(folder, out_folder):
    print(f"Processing {folder}...")
    # each recording
    for f in tqdm(os.listdir(os.path.join(root, folder))):
        print(f"Processing {folder}/{f}...")
        record = pickle.load(open(os.path.join(root, folder, f), "rb"))
        """
        {'FP1-F7': array([-145.93406593,    0.1953602 ,    0.1953602 , ...,  -11.52625153, -2.93040293,   19.34065934]), 
         'F7-T7': array([-104.51770452,    0.1953602 ,    0.1953602 , ...,   23.63858364, 27.54578755,   30.67155067]), 
         'T7-P7': array([-42.78388278,   0.1953602 ,   0.1953602 , ...,  48.64468864, 45.12820513,  34.57875458]), 
        'P7-O1': array([-33.01587302,   0.1953602 ,   0.1953602 , ..., -17.77777778, -20.51282051, -25.59218559]), 
       'FP1-F3': array([-170.94017094,    0.1953602 ,    0.1953602 , ...,  -34.96947497, -25.98290598,    0.1953602 ]), 
        'F3-C3': array([-110.76923077,    0.1953602 ,    0.1953602 , ...,   38.0952381 , 48.64468864,   50.20757021]), 
         'C3-P3': array([11.91697192,  0.1953602 ,  0.1953602 , ..., 40.04884005, 33.7973138 , 25.98290598]), 
       'P3-O1': array([-56.45909646,   0.1953602 ,   0.1953602 , ...,   0.97680098, -6.44688645, -16.60561661]), 
        'FP2-F4': array([-139.29181929,    0.1953602 ,    0.1953602 , ...,   -2.14896215, -2.14896215,   -0.58608059]), 
         'F4-C4': array([-1.36752137,  0.1953602 ,  0.1953602 , ...,  1.75824176, 2.93040293,  7.22832723]), 
        'C4-P4': array([63.88278388,  0.1953602 ,  0.1953602 , ..., 16.996337  , 23.63858364, 25.59218559]), 
       'P4-O2': array([-14.26129426,   0.1953602 ,   0.1953602 , ..., -13.08913309, -8.00976801, -13.47985348]), 
        'FP2-F8': array([-2.67838828e+02,  1.95360195e-01,  1.95360195e-01, ..., 6.83760684e+00,  6.05616606e+00,  6.44688645e+00]), 
        'F8-T8': array([ 57.24053724,   0.1953602 ,   0.1953602 , ...,  -2.53968254,  -9.96336996, -12.6984127 ]), 
        'T8-P8': array([44.73748474,  0.1953602 ,  0.1953602 , ..., 16.996337  , 22.46642247, 26.37362637]), 
       'P8-O2': array([ 74.82295482,   0.1953602 ,  -0.1953602 , ..., -17.38705739, -1.75824176,  -2.53968254]), 
        'FZ-CZ': array([-106.08058608,    0.1953602 ,    0.1953602 , ...,   24.81074481, 28.71794872,   28.71794872]), 
         'CZ-PZ': array([84.59096459,  0.1953602 ,  0.1953602 , ..., 18.94993895, 20.51282051, 18.16849817]), 
       'P7-T7': array([ 43.17460317,   0.1953602 ,   0.1953602 , ..., -48.25396825, -44.73748474, -34.18803419]), 
       'T7-FT9': array([-57.24053724,   0.1953602 ,   0.1953602 , ..., -11.91697192,  -3.71184371,   2.14896215]), 
        'FT9-FT10': array([-2.64713065e+02,  1.95360195e-01,  5.86080586e-01, ..., 9.76800977e-01, -1.58241758e+01, -2.94993895e+01]), 
        'FT10-T8': array([ 94.74969475,   0.1953602 ,   0.1953602 , ...,  -7.22832723, -10.35409035, -13.47985348]), 
       'T8-P8-2': array([44.73748474,  0.1953602 ,  0.1953602 , ..., 16.996337  , 22.46642247, 26.37362637]), 
       'metadata': {'seizures': 0, 'times': [], 'channels': ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-2']}}
        """
        signal = []
        for channel in channels:
            if channel in record:
                signal.append(record[channel])
            else:
                raise ValueError(f"Channel {channel} not found in record {record}")
        signal = np.array(signal)

        if "times" in record["metadata"]:
            seizure_times = record["metadata"]["times"]
        else:
            seizure_times = []

        # split the signal into segments on the second dimension by SAMPLING_RATE * 10 seconds
        for i in range(0, signal.shape[1], SAMPLING_RATE * 10):
            segment = signal[:, i : i + 10 * SAMPLING_RATE]
            if segment.shape[1] == 10 * SAMPLING_RATE:
                # judge whether the segment contains seizures
                label = 0

                for seizure_time in seizure_times:
                    if (
                        i < seizure_time[0] < i + 10 * SAMPLING_RATE
                        or i < seizure_time[1] < i + 10 * SAMPLING_RATE
                    ):
                        label = 1
                        break

                # save the segment
                pickle.dump(
                    {"X": segment, "y": label},
                    open(
                        os.path.join(out_folder, f"{f.split('.')[0]}-{i}.pkl"),
                        "wb",
                    ),
                )

        for idx, seizure_time in enumerate(seizure_times):
            for i in range(
                max(0, seizure_time[0] - SAMPLING_RATE),
                min(seizure_time[1] + SAMPLING_RATE, signal.shape[1]),
                5 * SAMPLING_RATE,
            ):
                segment = signal[:, i : i + 10 * SAMPLING_RATE]
                label = 1
                # save the segment
                pickle.dump(
                    {"X": segment, "y": label},
                    open(
                        os.path.join(
                            out_folder, f"{f.split('.')[0]}-s-{idx}-add-{i}.pkl"
                        ),
                        "wb",
                    ),
                )


# parallel parameters
folders = os.listdir(root)
out_folders = []
for folder in folders:
    if folder in test_pats:
        out_folder = os.path.join(out, "test")
    elif folder in val_pats:
        out_folder = os.path.join(out, "val")
    else:
        out_folder = os.path.join(out, "train")

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    out_folders.append(out_folder)

# process in parallel
with mp.Pool(mp.cpu_count()) as pool:
    res = pool.starmap(sub_to_segments, zip(folders, out_folders))
