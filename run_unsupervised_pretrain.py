import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import UnsupervisedPretrain
from utils import UnsupervisedPretrainLoader, collate_fn_unsupervised_pretrain

     
class LitModel_supervised_pretrain(pl.LightningModule):
    def __init__(self, args, save_path):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.T = 0.2
        self.model = UnsupervisedPretrain(emb_size=256, heads=8, depth=4, n_channels=18) # 16 for PREST (resting) + 2 for SHHS (sleeping)
        
    def training_step(self, batch, batch_idx):

        # store the checkpoint every 5000 steps
        if self.global_step % 2000 == 0:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )

        prest_samples, shhs_samples = batch
        contrastive_loss = 0

        if len(prest_samples) > 0:
            """
            For prest
            """
            prest_masked_emb, prest_samples_emb = self.model(prest_samples, 0)

            # L2 normalize
            prest_samples_emb = F.normalize(prest_samples_emb, dim=1, p=2)
            prest_masked_emb = F.normalize(prest_masked_emb, dim=1, p=2)
            N = prest_samples.shape[0]

            # representation similarity matrix, NxN
            logits = torch.mm(prest_samples_emb, prest_masked_emb.t()) / self.T
            labels = torch.arange(N).to(logits.device)
            contrastive_loss += F.cross_entropy(logits, labels, reduction="mean")

        """
        For shhs
        """
        shhs_masked_emb, shhs_samples_emb = self.model(shhs_samples, 16)

        # For shhs
        shhs_samples_emb = F.normalize(shhs_samples_emb, dim=1, p=2)
        shhs_masked_emb = F.normalize(shhs_masked_emb, dim=1, p=2)
        N = shhs_samples_emb.shape[0]

        # representation similarity matrix, NxN
        logits = torch.mm(shhs_samples_emb, shhs_masked_emb.t()) / self.T
        labels = torch.arange(N).to(logits.device)
        contrastive_loss += F.cross_entropy(logits, labels, reduction="mean")

        self.log("train_loss", contrastive_loss)
        return contrastive_loss


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

    # define the (seizure) data loader
    root_prest = "/srv/local/data/IIIC_data/5M_IIIC_data/processed/s7n16"
    root_shhs = "/srv/local/data/SHHS/processed"
    loader = UnsupervisedPretrainLoader(root_prest, root_shhs)
    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn_unsupervised_pretrain,
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
    save_path = f"log-pretrain/{N_version}-unsupervised/checkpoints"
    
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
    args = parser.parse_args()
    print (args)

    pretrain(args)
    
    
    