import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

class RedPatchDataset(Dataset):
    def __init__(self, config, num_samples=100):
        self.num_samples = num_samples
        self.image_size = config.img_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_patches_x = self.image_size // self.patch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a 6x6 RGB image (3 channels)
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Create a 6x6 mask (single channel)
        mask = torch.zeros(1, self.image_size, self.image_size)

        # Choose a random patch (out of the 4 available 3x3 patches)
        patch_idx = random.randint(0, self.num_patches - 1)
        row_start = (patch_idx // self.num_patches_x) * self.patch_size
        col_start = (patch_idx % self.num_patches_x) * self.patch_size
        
        # Set the randomly chosen patch to red (RGB: [1, 0, 0])
        image[0, row_start:row_start+self.patch_size, col_start:col_start+self.patch_size] = 1  # Red channel
        image[1, row_start:row_start+self.patch_size, col_start:col_start+self.patch_size] = 0  # Red channel
        image[2, row_start:row_start+self.patch_size, col_start:col_start+self.patch_size] = 0  # Red channel

        # Set the corresponding area in the mask to 1
        mask[0, row_start:row_start+self.patch_size, col_start:col_start+self.patch_size] = 1
        rand_x = torch.randint(row_start, row_start+self.patch_size, size=(1,))
        rand_y = torch.randint(col_start, col_start+self.patch_size, size=(1,))
        
        random_point = torch.cat([rand_x, rand_y])[None, None] / self.image_size
        top_left = (row_start, col_start)
        top_right = (row_start, col_start + self.patch_size)
        bottom_left = (row_start + self.patch_size, col_start)
        bottom_right = (row_start + self.patch_size, col_start + self.patch_size)
    
        contour = torch.tensor([top_left, top_right, bottom_left, bottom_right])[None] / self.image_size
        return contour.float(), image.float()
    
if __name__ == "__main__":
    # Example usage
    dataset = RedPatchDataset(num_samples=10)
    image, mask = dataset[0]

    print("Image Shape:", image.shape)
    print("Mask Shape:", mask.shape)
    print(image)
    print(mask)
