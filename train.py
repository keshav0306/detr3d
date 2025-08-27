import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from datasets_all.fs_dataset import FSDataset
from datasets_all.nusc3d import NuScenesDataset, collate_fn
from lit_model import LITDetModel
import yaml
from torch.utils.data import WeightedRandomSampler
import numpy as np

L.seed_everything(2024)
class Config:
    def __init__(self, config):
        self.config = config
        for k, v in self.config.items():
            self.__setattr__(k,  v)

config = "configs/red.yaml"
with open(config, "r") as f:
    config = Config(yaml.safe_load(f))

dataset_class = {"nuscenes" : NuScenesDataset}

train_dataset = dataset_class[config.dataset_type](config.dataset_path)
val_dataset = dataset_class[config.dataset_type](config.dataset_path, is_train="val")

trainloader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, collate_fn=collate_fn, shuffle=False)#, sampler=train_sampler)
valloader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, collate_fn=collate_fn, shuffle=False)

model = LITDetModel(config)
logger = WandbLogger(name="nuscenes_train", project="detr3d", config=config)
# logger = CSVLogger("logs", name="diff_fs")
checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=config.ckpt_dir, filename=f"best_loss_{config.backbone}", mode='min')

trainer = L.Trainer(devices=[0, 1, 2], max_epochs=1000, callbacks=[checkpoint_callback], logger=[logger],\
    strategy='ddp_find_unused_parameters_true', detect_anomaly=True)
trainer.fit(model, trainloader, valloader)