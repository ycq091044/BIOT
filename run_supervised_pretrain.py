import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import SupervisedPretrain
from utils import EEGSupervisedPretrainLoader, focal_loss, BCE, collate_fn_supervised_pretrain

     
class LitModel_supervised_pretrain(pl.LightningModule):
    def __init__(self, args, save_path):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.model = SupervisedPretrain(emb_size=256, heads=8, depth=4, n_channels=16)
        # load the pre-trained SHHS+PREST model
        self.model.biot.load_state_dict(torch.load(args.pretrained_model_path))
        
    def training_step(self, batch, batch_idx):

        # store the checkpoint every 5000 steps
        if self.global_step % 2000 == 0:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )

        (tuev_x, tuev_y), (chb_mit_x, chb_mit_y), (iiic_x, iiic_y), (tuab_x, tuab_y) = batch
        
        # for TUEV
        if len(tuev_y) > 0:
            convScore = self.model(tuev_x, task="tuev")
            loss1 = nn.CrossEntropyLoss()(convScore, tuev_y)
        else:
            loss1 = 0
            
        # for CHB-MIT
        if len(chb_mit_y) > 0:
            convScore = self.model(chb_mit_x, task="chb-mit")
            loss2 = focal_loss(convScore, chb_mit_y) * 200
        else:
            loss2 = 0   

        # for IIIC
        if len(iiic_y) > 0:
            convScore = self.model(iiic_x, task="iiic-seizure")
            loss3 = nn.CrossEntropyLoss()(convScore, iiic_y)
        else:
            loss3 = 0
            
        # for TUAB
        if len(tuab_y) > 0:
            convScore = self.model(tuab_x, task="tuab")
            loss4 = BCE(convScore, tuab_y)
        else:
            loss4 = 0
                
        self.log("loss_tuev", loss1)
        self.log("loss_chb_mit", loss2)
        self.log("loss_iiic", loss3)
        self.log("loss_tuab", loss4)
        self.log("loss", loss1 + loss2 + loss3 + loss4)
        return loss1 + loss2 + loss3 + loss4


    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        # set learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.3
        )

        return [optimizer], [scheduler]

    
def prepare_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # for TUEV
    print ("load data from TUEV")
    tuev_root = "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf"
    
    train_files = os.listdir(os.path.join(tuev_root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    val_sub = np.random.choice(train_sub, size=int(len(train_sub)*0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]
    print ('train files:', len(train_files))
    TUEV_data = (os.path.join(tuev_root, "processed_train"), train_files)
    
    # for CHB-MIT
    print ("load data from CHB-MIT")
    chb_mit_root = "/srv/local/data/physionet.org/files/chbmit/1.0.0/clean_segments"
    
    train_files = os.listdir(os.path.join(chb_mit_root, "train"))
    print ('train files:', len(train_files))
    CHB_MIT_data = (os.path.join(chb_mit_root, "train"), train_files)
    
    # for IIIC seizure
    print ("load data from IIIC seizure")
    train_pat_map = pickle.load(
        open("/home/chaoqiy2/github/LEM/mgh-seizure/data/train_pat_map_seizure.pkl", "rb")
    )
    train_X, train_Y = [], []
    for i, (_, (X, Y)) in enumerate(train_pat_map.items()):
        valid_idx = np.where((np.sum(np.array(Y) == np.max(Y, 1, keepdims=True), 1) == 1))[0]
        X = [X[item] for item in valid_idx]
        Y = [Y[item] for item in valid_idx]
        train_X += X
        train_Y += Y
    print ('train files:', len(train_X))
    IIIC_data = (train_X, train_Y)
    
    # for TUAB
    print ("load data from TUAB")
    tuab_root = "/srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf/processed"
    
    train_files = os.listdir(os.path.join(tuab_root, "train"))
    print ('train files:', len(train_files))
    TUAB_data = (os.path.join(tuab_root, "train"), train_files)
    
    
    train_loader = torch.utils.data.DataLoader(
        EEGSupervisedPretrainLoader(TUEV_data, CHB_MIT_data, IIIC_data, TUAB_data), 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        persistent_workers=True, 
        collate_fn=collate_fn_supervised_pretrain,
    )
    
    return train_loader
 
 
def pretrain(args):
    
    # get data loaders
    train_loader = prepare_dataloader(args)
    
    # define the trainer
    N_version = (
        len(os.listdir(os.path.join("log-pretrain"))) + 1
    )
    # define the model
    save_path = f"log-pretrain/{N_version}-supervised/checkpoints"
    
    model = LitModel_supervised_pretrain(args, save_path)
    
    logger = TensorBoardLogger(
        save_dir="/home/chaoqiy2/github/LEM",
        version=f"{N_version}/checkpoints",
        name="log-pretrain",
    )
    trainer = pl.Trainer(
        devices=[2],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
    )

    # train the model
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="number of workers")
    parser.add_argument("--pretrained_model_path", type=str, default="best", help="checkpoint path")
    args = parser.parse_args()
    print (args)

    pretrain(args)
    
    
    