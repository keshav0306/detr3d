import torch
import numpy as np
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm
import pickle

class NuScenesDataset(Dataset):
    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
        # self.nusc_can = NuScenesCanBus(dataroot=dataroot)
        
        self.train_scenes = self.get_scenes(0)
        self.val_scenes = self.get_scenes(1)
        self.test_scenes = self.get_scenes(2)
        
    def __len__(self):
        return len(self.nusc.sample)
    
    def get_scenes(self, is_train):
        # filter by scene split
        split = {'v1.0-trainval': {0: 'train', 1: 'val', 2: 'test'},
                 'v1.0-mini': {0: 'mini_train', 1: 'mini_val'},}[
            self.nusc.version
        ][is_train]

        blacklist = [419]# + self.nusc_can.can_blacklist  # # scene-0419 does not have vehicle monitor data
        blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]

        scenes = create_splits_scenes()[split][:]
        for scene_no in blacklist:
            if scene_no in scenes:
                scenes.remove(scene_no)

        return scenes

    def __getitem__(self, idx):
        # idx = 3009
        # idx = 13
        sample_token = self.nusc.sample[idx]['token']
        camera_channel = 'CAM_FRONT'
        
        sample = self.nusc.get("sample", sample_token)
        scene_token = sample["scene_token"]
        scene = self.nusc.get("scene", scene_token)
        scene_number = scene["name"]
        
        suffix = "train"
        if(scene_number in self.val_scenes):
            suffix = "val"
        elif (scene_number in self.test_scenes):
            suffix = "test"
        
        cam_token = sample['data']['CAM_FRONT']
        cam_record = self.nusc.get('sample_data', cam_token)
        cam_path = self.nusc.get_sample_data_path(cam_token)
        cs_record = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        
        data_path, box_list, cam_intrinsic = self.nusc.get_sample_data(cam_token)
        # print(data_path)
        # print(box_list)
        # print(cam_intrinsic)
        
        # im = np.asarray(Image.open(data_path))
        # for box in box_list:
        #     box.render_cv2(im, view=cam_intrinsic, normalize=True)
        #     print(box)
        # cv2.imwrite("debug.png", im)
        # print(boxes[0])
        
        filename = cam_path.split("/")[-1]
        data = {"data_path": data_path, "box_list" : box_list, "cam_intrinsic": cam_intrinsic}

        with open(f"/ssd_scratch/cvit/keshav/nuscenes/{suffix}_3d/{filename.split('.')[0]}.pkl", "wb") as f:
            pickle.dump(data, f)

        return cam_intrinsic


dataset = NuScenesDataset("/ssd_scratch/cvit/keshav/nuscenes/")
dataLoader = DataLoader(dataset, batch_size=16, num_workers=8)
import os
os.makedirs("/ssd_scratch/cvit/keshav/nuscenes/train_3d/", exist_ok=True)
os.makedirs("/ssd_scratch/cvit/keshav/nuscenes/val_3d/", exist_ok=True)
os.makedirs("/ssd_scratch/cvit/keshav/nuscenes/test_3d/", exist_ok=True)
for batch in tqdm(dataLoader):
    batch
    # exit(0)