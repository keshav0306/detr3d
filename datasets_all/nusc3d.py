import torch
import os
import numpy as np
import glob
import pickle
import cv2
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import Box
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix

class NuScenesDataset(Dataset):
    def __init__(self, path, is_train='train'):
        split = is_train
        self.path = path
        if(split == "train"):
            self.path = os.path.join(self.path, "train_3d")
        elif(split == "val"):
            self.path = os.path.join(self.path, "val_3d")
        else:
            self.path = os.path.join(self.path, "test_3d")
        
        all_files = glob.glob(self.path + "/*.pkl")
        self.data = []
        for file in all_files:
            with open(file, "rb") as f:
                data = pickle.load(f)
                self.data.append(data)
        self.split = split
        
    def __len__(self):
        return len(self.data)

    def visualize_gt(self, im, box_list, cam_intrinsic, name=None):
        for box in box_list:
            box.render_cv2(im, view=cam_intrinsic, normalize=True)
        if(name is None):
            name = "debug.png"
        cv2.imwrite(name, im)

    def __getitem__(self, idx):
        # idx = 14
        data = self.data[idx]
        data_dict = {}
        img = cv2.imread(data["data_path"])
        img = cv2.resize(img, (256, 256))
        data_dict["img"] = img.transpose(2, 0, 1) # / 255.0 -> Handled in the backbone code
        
        boxes = data["box_list"]
        # self.visualize_gt(img, boxes, data["cam_intrinsic"], name=f"/ssd_scratch/cvit/keshav/viz/{idx}.jpg")
        xyz = []
        yaw = []
        wlh = []
        
        for box in boxes:
            if("car" in box.name):
                xyz.append(box.center)
                yaw.append(box.orientation.yaw_pitch_roll[0])
                wlh.append(box.wlh)

        if(len(xyz)):
            data_dict["xyz"] = np.array(xyz)
            data_dict["yaw"] = np.array(yaw)
            data_dict["wlh"] = np.array(wlh)
        else:
            data_dict["xyz"] = np.zeros((0, 3))
            data_dict["yaw"] = np.zeros((0))
            data_dict["wlh"] = np.zeros((0, 3))
        data_dict["K"] = data["cam_intrinsic"]
        
        # vis_img = cv2.resize(img, (256, 256))
        # fig, ax = plt.subplots(1, 2, squeeze=False)
        # ax[0, 0].imshow(vis_img)
        # points = []
        # try:
        #     for box in boxes:
        #         points.append((box.center[0], box.center[1]))
        #     points = np.array(points)
        #     ax[0, 1].scatter(points[:, 0], points[:, 1])
        #     # add text alongside the scatter of yaw
        #     for i, txt in enumerate(yaw):
        #         ax[0, 1].annotate(f"{(txt*57):.2f}", (points[i, 0], points[i, 1]))
        #     ax[0, 1].set_xlim(-20, 20)
        #     ax[0, 1].set_ylim(-10, 50)
        #     plt.savefig(f"{idx}.png")
        #     plt.close()
        # except:
        #     pass
        
        return data_dict
    
def collate_fn(batch):
    max_elem = max([len(elem['xyz']) for elem in batch])
    all_xyz = []
    all_yaw = []
    all_wlh = []
    all_img = []
    all_cam_intrinsic = []
    all_mask = []
    
    for elem in batch:
        all_xyz.append(np.pad(elem['xyz'], ((0, max_elem - len(elem['xyz'])), (0, 0)), mode='constant'))
        all_yaw.append(np.pad(elem['yaw'], (0, max_elem - len(elem['yaw'])), mode='constant'))
        all_wlh.append(np.pad(elem['wlh'], ((0, max_elem - len(elem['wlh'])), (0, 0)), mode='constant'))
        all_mask.append(len(elem['xyz']))
        all_img.append(elem['img'])
        all_cam_intrinsic.append(elem['K'])
    
    new_dict = {}
    new_dict["xyz"] = torch.from_numpy(np.array(all_xyz)).float()
    new_dict["yaw"] = torch.from_numpy(np.array(all_yaw)).float()
    new_dict["wlh"] = torch.from_numpy(np.array(all_wlh)).float()
    new_dict["img"] = torch.from_numpy(np.array(all_img)).float()
    new_dict["mask"] = torch.from_numpy(np.array(all_mask))
    new_dict["K"] = torch.from_numpy(np.array(all_cam_intrinsic)).float()
    
    return new_dict
    
if __name__ == "__main__":
    dataset = NuScenesDataset("/ssd_scratch/cvit/keshav/nuscenes", is_train='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=collate_fn)

    for batch in dataloader:
        images = batch['img']
        xyz = batch['xyz']
        # print(xyz)
        