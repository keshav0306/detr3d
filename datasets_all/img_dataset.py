import numpy as np
import cv2
import cv2
import tqdm
import torch
import sys
from .basic_dataset import BasicDataset

def transform_ego(ego_locs, locs, oris, bbox, typs, ego_ori, T=11):

    ego_loc = ego_locs[0]

    keys = sorted(list(locs.keys()))
    locs = np.array([locs[k] for k in keys]).reshape(-1,T,2)
    oris = np.array([oris[k] for k in keys]).reshape(-1,T)
    bbox = np.array([bbox[k] for k in keys]).reshape(-1,T,2)
    typs = np.array([typs[k] for k in keys]).reshape(-1,T)

    R = [[np.sin(ego_ori),np.cos(ego_ori)],[-np.cos(ego_ori),np.sin(ego_ori)]] 
    
    locs = (locs-ego_loc) @ R
    ego_locs = (ego_locs-ego_loc) @ R
    oris = oris - ego_ori

    return ego_locs, locs, oris, bbox, typs


class RGBDataset(BasicDataset):
    def __init__(self, config_path, is_train=True):
        super().__init__(config_path, is_train=is_train)
    
    # def __len__(self):
        # return 1
    
    def preprocess(self, img):
        return img / 255

    def __getitem__(self, idx):

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]

        imgs = []
        for t in [0, 1]:
            rgb = self.__class__.load_img(lmdb_txn, f'rgb_2', index + t)
            imgs.append(self.preprocess(rgb))
        # cv2.imwrite(f"rgb.png", np.hstack(imgs) * 255)
        return torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float()

if __name__ == '__main__':
    dataset = RGBDataset('lav.yaml', is_train=False)

    for t in tqdm.tqdm(range(1000)):
        rgb = dataset[t]
        # for i in range(2):
            # cv2.imwrite(f"rgb{i}.png", rgb[i])