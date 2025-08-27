import torch
import numpy as np
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class NuScenesDataset(Dataset):
    def __init__(self, dataroot, is_train=True):
        self.dataroot = dataroot
        self.cam_front_images = glob.glob(dataroot + "/samples/CAM_FRONT/*.jpg")
        self.fs_masks = glob.glob(dataroot + "/samples/FS/*.jpg")
        self.meta_data = glob.glob(dataroot + "/fs_meta/*.npz")
        self.num_contour_points = 50
        self.is_train = is_train
        if(is_train):
            self.num_images = int(0.8 * len(self.cam_front_images))
            self.cam_front_images = self.cam_front_images[:self.num_images]
            self.fs_masks = self.fs_masks[:self.num_images]
            self.meta_data = self.meta_data[:self.num_images]
        else:
            self.num_images = len(self.cam_front_images) - int(0.8 * len(self.cam_front_images))
            self.cam_front_images = self.cam_front_images[self.num_images:]
            self.fs_masks = self.fs_masks[self.num_images:]
            self.meta_data = self.meta_data[self.num_images:]
        self.count = 0

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img = cv2.imread(self.cam_front_images[idx])/255
        img = cv2.resize(img, (256, 256))
        fs_mask = cv2.imread(self.fs_masks[idx], cv2.IMREAD_GRAYSCALE)
        fs_mask[fs_mask >= 1] = 1
        fs_mask[fs_mask < 1] = 0
        fs_mask = cv2.resize(fs_mask, (256, 256))
        # meta_data = np.load(self.meta_data[idx])
        
        contours, hierarchy = cv2.findContours(fs_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        all_contours = []
        
        # output_image = img.copy()
        all_len = []
        for contour in contours:
            contour_interp = self.interpolate_contour_points(contour[:, 0], self.num_contour_points)
            all_len.append(len(contour[:, 0]))
            all_contours.append(contour_interp)
            # for point in contour_interp:
            #     x, y = point.astype(np.int32)  # Get the x, y coordinates
            #     cv2.circle(output_image, (x, y), radius=1, color=(0, 1, 0), thickness=1)
        
        # img[fs_mask == 1] = [1, 0, 0]
        
        valid = True
        if(len(all_contours) == 0):
            contour = np.ones((self.num_contour_points, 2)) * -2
            valid = False
        else:
            max_index = np.array(all_len).argmax()
            contour = all_contours[max_index]
            contour = contour / 256
            contour = contour * 2 - 1
            
        img = img.transpose(2, 0, 1).astype(np.float32)
        # sem = cv2.resize(sem, (64, 64)).astype(np.float32).transpose(2, 0, 1)
        data = {"img" : img, "mask": fs_mask, "contour": contour[None].astype(np.float32)}
        # data['vis'] = output_image * 255
        data['valid'] = valid # for now
        # data['angle'] = angle
        data['idx'] = idx
        # cv2.imwrite(f"/ssd_scratch/cvit/keshav/nusc_contour_vis/{idx}.png", output_image*255)
        
        return data
        # return fs_mask
    
    def interpolate_contour_points(self, contour, target_num_points):
        num_points = len(contour)
        # if num_points >= target_num_points:
        #     return contour
        indices = np.linspace(0, num_points - 1, target_num_points, dtype=np.float32)
        
        contour_interp = np.zeros((target_num_points, 2), dtype=np.float32)
        for i in range(2):
            contour_interp[:, i] = np.interp(indices, np.arange(num_points), contour[:, i])
        
        return contour_interp


if __name__ == "__main__":
    dataset = NuScenesDataset("/ssd_scratch/cvit/keshav/nuscenes/")
    dataLoader = DataLoader(dataset, batch_size=64)
    for batch in tqdm(dataLoader):
        batch
        # exit(0)
        print(dataset.count)