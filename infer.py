import torch
import numpy as np
import cv2
import lightning as L
from tqdm import tqdm
from datasets_all.nusc3d import NuScenesDataset, collate_fn
import yaml
import glob
import os
from lit_model import LITDetModel

class Config:
    def __init__(self, config):
        self.config = config
        for k, v in self.config.items():
            self.__setattr__(k,  v)

config = "configs/red.yaml"

with open(config, "r") as f:
    config = Config(yaml.safe_load(f))
    
dataset_class = {"nuscenes" : NuScenesDataset}

val_dataset = dataset_class[config.dataset_type](config.dataset_path, is_train="train")
valloader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=32, num_workers=4, collate_fn=collate_fn)
model = LITDetModel.load_from_checkpoint("/ssd_scratch//cvit/keshav/detr3d_ckpts/best_loss_resnet18-v27.ckpt", config=config)
count = 0
# imgs = glob.glob("/home2/keshav06/transfuser/*.png")
# os.makedirs("/ssd_scratch/cvit/keshav/det/", exist_ok=True)
# imgs = [cv2.imread(f)[None] for f in imgs]
# imgs = torch.from_numpy(np.concatenate(imgs)).permute(0, 3, 1, 2).cuda()

with torch.no_grad():
    for batch in tqdm(valloader):
        # img = val_dataset[i][0].permute(1, 2, 0).cpu().numpy()
        data = batch
        K, img_ = data['K'].cuda(), data['img'].cuda()
        
        pred_boxes = model.model.infer(img_, K)
        vis_imgs = model.visualize(img_, pred_boxes, K)
        for i in tqdm(range(len(img_))):
            cv2.imwrite(f"/ssd_scratch/cvit/keshav/det/{count}.png", vis_imgs[i])
            count += 1