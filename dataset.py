import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class SurfaceDataset(Dataset):

    def __init__(self,image_paths,mask_paths,transform=None):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):

        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(256,256))

        mask = cv2.imread(self.mask_paths[idx],cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,(256,256))

        if self.transform:

            augmented = self.transform(image=image,mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]


        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0


        image = np.transpose(image,(2,0,1))
        mask = np.expand_dims(mask,axis=0)


        image = torch.tensor(image,dtype=torch.float32)
        mask = torch.tensor(mask,dtype=torch.float32)

        return image, mask