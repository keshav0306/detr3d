import numpy as np
import cv2
import torch
import sys
import tqdm
import matplotlib.pyplot as plt
sys.path.append("/home2/keshav06/LAV/")
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

class BEVDataset(BasicDataset):
    def __init__(self, config_path, is_train=True):
        super().__init__(config_path, is_train=is_train)
        self.max_other = 20
    
    def get_prev_bevs(self, lmdb_txn, index):
        context = self.num_prev_timesteps
        prev_bevs = []
        for t, i in enumerate(range(index-context,index)):
            if(i < 0):
                prev_bevs.append(np.zeros((4, 320, 320)))
                continue
            bev = self.__class__.load_bev(lmdb_txn, i, channels=[0,1,2,6])
            bev = (bev>0).astype(np.uint8).transpose(2,0,1)
            prev_bevs.append(bev)
        
        return prev_bevs

    def __getitem__(self, idx):

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]

        rgb1 = self.__class__.load_img(lmdb_txn, 'rgb_2', index)
        # rgb2 = self.__class__.load_img(lmdb_txn, 'rgb_3', index)
        # sem1 = self.__class__.load_img(lmdb_txn, 'sem_2', index)
        # sem2 = self.__class__.load_img(lmdb_txn, 'sem_3', index)
        
        # BEV images
        bev = self.__class__.load_bev(lmdb_txn, index, channels=[0,1,2,6])
        bev = (bev>0).astype(np.uint8).transpose(2,0,1)
        prev_bevs = self.get_prev_bevs(lmdb_txn, index)
        prev_bevs.append(bev)
        bevs = np.stack(prev_bevs)

        # rgb = np.concatenate([rgb1, rgb2], axis=1)
        # sem = np.concatenate([sem1, sem2], axis=1)

        # rgb = self.augmenter(images=rgb[...,::-1][None])[0]
        # sem = filter_sem(sem, self.seg_channels)

        # Vehicle locations/orientations
        ego_id, ego_locs, ego_oris, ego_bbox, msks, locs, oris, bbox, typs = self.__class__.filter(
            lmdb_txn, index,
            max_pedestrian_radius=self.max_pedestrian_radius,
            max_vehicle_radius=self.max_vehicle_radius,
            T=self.num_plan)

        # Normalize coordinates to ego frame
        ego_locs, locs, oris, bbox, typs = transform_ego(ego_locs, locs, oris, bbox, typs, ego_oris[0], self.num_plan+1)

        cmd = int(self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.uint8))
        nxp = self.__class__.access('nxp', lmdb_txn, index, 1).reshape(2)
        
        # bbox_other = np.zeros((self.max_other, self.num_plan+1, 2))
        # oris_other = np.zeros((self.max_other, self.num_plan+1))
        # locs_other = np.zeros((self.max_other, self.num_plan+1, 2))
        # mask_other = np.zeros(self.max_other)
        # bbox_other[:bbox.shape[0]] = bbox[:self.max_other]
        # locs_other[:locs.shape[0]] = locs[:self.max_other]
        # oris_other[:oris.shape[0]] = oris[:self.max_other]
        # mask_other[:mask_other.shape[0]] = 1
        rgb1 = cv2.resize(rgb1, (256, 256))
        return bevs.astype(np.float32), rgb1.astype(np.float32).transpose(2, 0, 1)/255, locs[1:], bbox[1:], oris[1:]

if __name__ == '__main__':

    dataset = BEVDataset('lav.yaml')
    import tqdm 
    PIXELS_PER_METER = 4
    PIXELS_AHEAD_VEHICLE = 120
    
    for i in tqdm.tqdm(range(3000)):
        bev, points = dataset[i]
        # for point in points:
            # x, y = points
            # cv2.circle(bev, (x, y), radius=2, color=(255, 255, 255), thickness=1)
        # cv2.imwrite(f"/ssd_scratch/cvit/keshav/imgs/{i:4d}.png", img)
        bev = bev.transpose(1, 2, 0)
        canvas = bev[..., 1] + bev[..., -1]
        print(points)
        for (px, py) in points:
            px, py = int(320/2 - px * PIXELS_PER_METER), int(320/2 - py * PIXELS_PER_METER + PIXELS_AHEAD_VEHICLE)
            print(px, py)
            cv2.circle(canvas, (px, py), 2, (1, 0, 1), -1)
        cv2.imwrite(f"bev_{i}.png", canvas * 255)
        break
