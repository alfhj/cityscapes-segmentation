
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import bz2
import os
import pickle
import numpy as np

h, w = 1024, 2048

transform_common = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((h//4, w//4), antialias=True),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomCrop()
    #transforms.RandomVerticalFlip(p=0.5),
    #transforms.RandomPerspective(p=0.5),
    #transforms.CenterCrop((1024, 1024)),
    #transforms.Resize((512, 512), antialias=True)
])

transform_img = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #transforms.Normalize(mean=means, std=stds),
])

"""
transform_img = A.Compose([
    #A.CenterCrop(round_to(h*0.8, 8), round_to(w*0.9, 8)), # crop away edge artefacts
    #A.HorizontalFlip(p=0.5),
    #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #A.Normalize(mean=means, std=stds, max_pixel_value=1),
    #A.Normalize(), # /255
    ToTensorV2(transpose_mask=True)
])

transform_val = A.Compose([
    A.Normalize(mean=[0.28766859, 0.32577001, 0.28470659], std=[0.17583184, 0.180675, 0.17738219]),
    ToTensorV2(transpose_mask=True)
])
"""


"""
transform_label = transforms.Compose([
    transforms.ToTensor(),
    #transforms.CenterCrop((1024, 1024)),
    #transform.Resize((512, 512))
    #transform.Resize((h//2, w//2)),
]) # You can define transformations for labels if necessary
"""

class MyDataset(Dataset):
    def __init__(self, name, images, ground_truth):

        self.name = name
        self.image_paths = images
        self.ground_truth = ground_truth
        self.cached_data = {}
        self.loaded_pickle = False
        #if os.path.exists("cached_data.pkl"):
        #    self._load_pickle()
        self._pixel_to_class = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 2, 3, 4, -1, -1, -1, 5, -1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._pixel_to_class) - 1)).astype("int32")

    def _class_to_index(self, mask):
        if mask.max() > 35:
            return mask
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if not idx in self.cached_data:
            raw_path = self.image_paths[idx]
            gt_path = raw_path.replace("leftImg8bit", "gtFine").replace(".png", "_color.png")
            raw_img = plt.imread(raw_path)
            ground_truth_img = plt.imread(gt_path)[:, :, :3]
            raw_img = transform_common(raw_img)
            ground_truth_img = transform_common(ground_truth_img)
            self.cached_data[idx] = (raw_img, ground_truth_img)
        else:
            raw_img, ground_truth_img = self.cached_data[idx]
        

        #transformed = self.transform_img(image=raw_img, mask=ground_truth_img)
        #raw_img = transformed["image"]
        #ground_truth_img = transformed["mask"]
        #raw_img = self.transform_common(raw_img)
        raw_img = transform_img(raw_img)
        #ground_truth_img = self.transform_common(ground_truth_img)


        #if not self.loaded_pickle and len(self.cached_data) == len(self.image_paths):
        #    self._save_pickle()
        
        return raw_img, ground_truth_img
    
    def _load_pickle(self):
        with bz2.BZ2File(f"{self.name}.pbz2", "r") as pkl:
            self.cached_data = pickle.load(pkl)
        self.loaded_pickle = True
     
    def _save_pickle(self):
        with bz2.BZ2File(f"{self.name}.pbz2", "w") as pkl:
            pickle.dump(self.cached_data, pkl)
        self.loaded_pickle = True
        

class CachedDataset(Dataset):
    def __init__(self):
        pass