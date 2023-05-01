# BIOT: Cross-data Biosignal Learning in the Wild
- This is a biosignal transformer model that is suitable for cross-data pre-training. 
- We also release the actual pre-trained EEG models.

## 0. Quick Start
- Understand the usage of each model (input and output)
```bash
# python run_example.py SPaRCNet
# python run_example.py ContraWR
# python run_example.py CNNTransformer
# python run_example.py FFCL
# python run_example.py STTransformer
python run_example.py BIOT
# python run_example.py BIOT-pretrain-PREST
# python run_example.py BIOT-pretrain-SHHS+PREST
# python run_example.py BIOT-pretrain-six-datasets 
```


## 1. Folder Structures
- **datasets/**: contains the dataset processing scripts for 
    - <u>The TUH Abnormal EEG Corpus (TUAB)</u>: 400K sample, 256Hz, 10 seconds per sample, 16 montages. https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    - <u>The TUH EEG Events Corpus (TUEV)</u>: 110K samples, 256Hz, 10 seconds per sample, 16 montages. https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    - <u>CHB-MIT</u>: 326K samples, 256Hz, 10 seconds per sample, 16 montages. https://physionet.org/content/chbmit/1.0.0/
    - <u>Sleep Heart Health Study (SHHS)</u>: 5M samples, 125Hz, 30 seconds per sample. https://sleepdata.org/datasets/shhs
- **models/**: contains the model scripts for 
    - <u>SPaRCNet [1]</u>: can refer to the implementation in https://pyhealth.readthedocs.io/en/latest/api/models/pyhealth.models.SparcNet.html 
    - <u>ContraWR [2]</u>: can refer to the implementation in https://pyhealth.readthedocs.io/en/latest/api/models/pyhealth.models.ContraWR.html or official github in https://github.com/ycq091044/ContraWR
    - <u>CNNTransformer</u>
    - <u>FFCL</u>
    - <u>STTransformer [3]</u>: can refer to https://github.com/eeyhsong/EEG-Transformer
    - <u>**Our BIOT Model**</u>: BIOTEncoder, BIOTClassifier, UnsupervisedPretrain, SupervisedPretrain

- **pretrained-models/**: contains the pretrained EEG models
> [1] Jing, J., Ge, W., Hong, S., Fernandes, M. B., Lin, Z., Yang, C., An, S., Struck, A. F., Herlopian, A., Karakis, I., et al. (2023). Development of expert-level classification of seizures and rhythmic and periodic patterns during eeg interpretation. Neurology.
[2] Yang, C., Xiao, D., Westover, M. B., and Sun, J. (2021). Self-supervised eeg representation learning for automatic sleep staging. arXiv preprint arXiv:2110.15278.
[3] Song, Y., Jia, X., Yang, L., and Xie, L. (2021). Transformer-based spatial-temporal feature learning
for eeg decoding. arXiv preprint arXiv:2106.11170.

## 2. EEG Pre-trained models (all re-sampled to 200Hz)
- `EEG-PREST-16-channels.ckpt`: pretrained model on 5 millions of resting EEG samples from Massachusetts General Hospital (MGH) EEG corpus. The sample size is 16 montages x 2000 time points. The 16 channels are "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2".
- `EEG-SHHS+PREST-18-channels.ckpt`: pretrained model on 5 millions samples from MGH EEG corpus and 5 millions of sleep EEG from SHHS. The SHHS sample size is 2 channels x 6000 time points. The 18 channels are the above 16 channels plus "C3-A2" and "C4-A1".
- `EEG-six-datasets-18-channels.ckpt`: pretrained on 5M MGH EEG samples, 5M SHHS, and the training sets of TUAB, TUEV, CHB-MIT, and IIIC Seizure (requested from [1]). The same 18 channels as above.

How to use the pretrained models:
- start wtih `run_example.py` to understand how to use these DL models 

To run your own pretrained models:
- use `run_supervised_pretrain.py` and `run_unsupervised_pretrain.py` for running the pre-training.


## 3. Performances on TUAB
- The first six models are trained from scratch. The last three models used the pre-trained BIOT.

| Models |   Balanced Accuracy |   AUC-PR    | AUROC |
|--------------------------|------------------------|------------------------|------------------------|
| SPaRCNet                 | 0.7896 ± 0.0018        | 0.8414 ± 0.0018        | 0.8676 ± 0.0012        |
| ContraWR                 | 0.7746 ± 0.0041        | 0.8421 ± 0.0104        | 0.8456 ± 0.0074        |
| CNN-Transformer          | 0.7777 ± 0.0022        | 0.8433 ± 0.0039        | 0.8461 ± 0.0013        |
| FFCL                     | 0.7848 ± 0.0038        | 0.8448 ± 0.0065        | 0.8569 ± 0.0051        |
| ST-Transformer           | 0.7966 ± 0.0023        | 0.8521 ± 0.0026        | 0.8707 ± 0.0019  |
| BIOT (vanilla)          | 0.7925 ± 0.0035 | 0.8707 ± 0.0087  | 0.8691 ± 0.0033        |
| BIOT (pre-trained on EEG-PREST-16-channels.ckpt) | 0.7907 ± 0.0050        | 0.8752 ± 0.0051        | 0.8730 ± 0.0021        |
| BIOT (pre-trained on EEG-SHHS+PREST-18-channels.ckpt) | **0.8019 ± 0.0021** | 0.8749 ± 0.0054   | 0.8739 ± 0.0019        |
| BIOT (pre-trained on EEG-six-datasets-18-channels.ckpt) | 0.7959 ± 0.0057  | **0.8792 ± 0.0023**  | **0.8815 ± 0.0043**  |

##### Reference Runs
```bash
python run_binary_supervised.py --dataset TUAB --in_channels 16 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 10 --batch_size 512 --model SPaRCNet
python run_binary_supervised.py --dataset TUAB --in_channels 16 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 10 --batch_size 512 --model ContraWR
python run_binary_supervised.py --dataset TUAB --in_channels 16 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 10 --batch_size 512 --model CNNTransformer
python run_binary_supervised.py --dataset TUAB --in_channels 16 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 10 --batch_size 512 --model FFCL
python run_binary_supervised.py --dataset TUAB --in_channels 16 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 10 --batch_size 512 --model BIOT
python run_binary_supervised.py --dataset TUAB --in_channels 16 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 10 --batch_size 512 --model BIOT --pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt
python run_binary_supervised.py --dataset TUAB --in_channels 18 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 10 --batch_size 512 --model BIOT --pretrain_model_path pretrained-models/EEG-SHHS+PREST-18-channels.ckpt
python run_binary_supervised.py --dataset TUAB --in_channels 18 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 10 --batch_size 512 --model BIOT --pretrain_model_path pretrained-models/EEG-six-datasets-18-channels.ckpt
```

## 4. Performance on TUEV
- The first six models are trained from scratch. The last three models used the pre-trained BIOT.

| Models                                                    | Balanced Accuracy | Cohen's Kappa          | Weighted F1          |
|-----------------------------------------------------------|-------------------|------------------|-----------------|
| SPaRCNet                                                  | 0.4161 ± 0.0262   | 0.4233 ± 0.0181  | 0.7024 ± 0.0104 |
| ContraWR                                                  | 0.4384 ± 0.0349   | 0.3912 ± 0.0237  | 0.6893 ± 0.0136 |
| CNN-Transformer                                           | 0.4087 ± 0.0161   | 0.3815 ± 0.0134  | 0.6854 ± 0.0293 |
| FFCL                                                      | 0.3979 ± 0.0104   | 0.3732 ± 0.0188  | 0.6783 ± 0.0120 |
| ST-Transformer                                            | 0.3984 ± 0.0228   | 0.3765 ± 0.0306  | 0.6823 ± 0.0190 |
| BIOT (Vanilla)                                            | 0.4682 ± 0.0125   | 0.4482 ± 0.0285  | 0.7085 ± 0.0184 |
| BIOT (pre-trained on PREST)                                  | 0.5207 ± 0.0285   | 0.4932 ± 0.0301  | 0.7381 ± 0.0169 |
| BIOT (pre-trained on PREST+SHHS)                             | 0.5149 ± 0.0292   | 0.4841 ± 0.0309  | 0.7322 ± 0.0196 |
| BIOT (pre-trained on CHB-MIT with 8 channels and 10s)       | 0.4123 ± 0.0087 | 0.4285 ± 0.0065 | 0.6989 ± 0.0015 |
| BIOT (pre-trained on CHB-MIT with 16 channels and 5s)       | 0.4218 ± 0.0117 | 0.4427 ± 0.0093 | 0.7147 ± 0.0058 |
| BIOT (pre-trained on CHB-MIT with 16 channels and 10s)      | 0.4344 ± 0.0065 | 0.4719 ± 0.0231 | 0.7280 ± 0.0126 |
| BIOT (pre-trained on IIIC seizure with 8 channels and 10s)  | 0.4956 ± 0.0552 | 0.4719 ± 0.0475 | 0.7214 ± 0.0220 |
| BIOT (pre-trained on IIIC seizure with 16 channels and 5s)  | 0.4894 ± 0.0189 | 0.4881 ± 0.0045 | 0.7348 ± 0.0056 |
| BIOT (pre-trained on IIIC seizure with 16 channels and 10s) | 0.4935 ± 0.0288 | **0.5316 ± 0.0176** | 0.7555 ± 0.0111 |
| BIOT (pre-trained on TUAB with 8 channels and 10s)          | 0.4980 ± 0.0384 | 0.4487 ± 0.0535 | 0.7044 ± 0.0365 |
| BIOT (pre-trained on TUAB with 16 channels and 5s)          | 0.4954 ± 0.0305 | 0.5053 ± 0.0079 | 0.7447 ± 0.0049 |
| BIOT (pre-trained on TUAB with 16 channels and 10s)         | 0.5256 ± 0.0348 | 0.5187 ± 0.0160 | **0.7504 ± 0.0102** |
| BIOT (pre-trained on 6 EEG datasets)                                | **0.5281 ± 0.0225**   | 0.5273 ± 0.0249  | 0.7492 ± 0.0082 |

##### Reference Runs
```bash
python run_multiclass_supervised.py --dataset TUEV --in_channels 16 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model SPaRCNet
python run_multiclass_supervised.py --dataset TUEV --in_channels 16 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model ContraWR
python run_multiclass_supervised.py --dataset TUEV --in_channels 16 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model CNNTransformer
python run_multiclass_supervised.py --dataset TUEV --in_channels 16 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model FFCL
python run_multiclass_supervised.py --dataset TUEV --in_channels 16 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model STTransformer
python run_multiclass_supervised.py --dataset TUEV --in_channels 16 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model BIOT
python run_multiclass_supervised.py --dataset TUEV --in_channels 16 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model BIOT --pretrain_model_path pretrained-models/EEG-PREST-16-channels.ckpt
python run_multiclass_supervised.py --dataset TUEV --in_channels 18 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model BIOT --pretrain_model_path pretrained-models/EEG-SHHS+PREST-18-channels.ckpt
python run_multiclass_supervised.py --dataset TUEV --in_channels 18 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 128 --model BIOT --pretrain_model_path pretrained-models/EEG-six-datasets-18-channels.ckpt
```


