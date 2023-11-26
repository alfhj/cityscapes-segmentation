
from collections import namedtuple
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import bz2
import os
import pickle

h, w = 1024, 2048

transform_common = transforms.Compose([
    #transforms.Resize((h//2, w//2), antialias=True),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomCrop()
    #transforms.RandomVerticalFlip(p=0.5),
    #transforms.RandomPerspective(p=0.5),
    #transforms.CenterCrop((1024, 1024)),
    #transforms.Resize((512, 512), antialias=True)
])

transform_img = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=0, std=255),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
        #self.cached_data = {}
        #self.loaded_pickle = False
        #if os.path.exists("cached_data.pkl"):
        #    self.load_pickle()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if True:# not idx in self.cached_data:
            raw_path = self.image_paths[idx]
            gt_path = raw_path.replace("leftImg8bit", "gtFine").replace(".png", "_color.png")
            raw_img = plt.imread(raw_path)
            ground_truth_img = plt.imread(gt_path)[:, :, :3]
            raw_img = transform_common(raw_img)
            ground_truth_img = transform_common(ground_truth_img)
            #raw_img.to("cuda")
            #ground_truth_img.to("cuda")
            #self.cached_data[idx] = (raw_img, ground_truth_img)
        else:
            raw_img, ground_truth_img = self.cached_data[idx]
        

        #transformed = self.transform_img(image=raw_img, mask=ground_truth_img)
        #raw_img = transformed["image"]
        #ground_truth_img = transformed["mask"]
        #raw_img = self.transform_common(raw_img)
        raw_img = transform_img(raw_img)
        #ground_truth_img = self.transform_common(ground_truth_img)


        #if not self.loaded_pickle and len(self.cached_data) == len(self.image_paths):
        #    self.save_pickle()
        
        return raw_img, ground_truth_img

    # def load_pickle(self):
    #     with bz2.BZ2File(f"{self.name}.pbz2", "r") as pkl:
    #         self.cached_data = pickle.load(pkl)
    #     self.loaded_pickle = True
     
    # def save_pickle(self):
    #     with bz2.BZ2File(f"{self.name}.pbz2", "w") as pkl:
    #         pickle.dump(self.cached_data, pkl)
    #     self.loaded_pickle = True
        
class CityscapesDataset(Dataset):
    def __init__(self, name, images, ground_truth, group_labels=False):
        self.name = name
        self.image_paths = images
        self.ground_truth = ground_truth
        self.num_classes = len(grouped_labels) if group_labels else len(labels)
        self.group_labels = group_labels
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raw_path = self.image_paths[idx]
        gt_path = raw_path.replace("leftImg8bit", "gtFine").replace(".png", "_labelIds.png")
        raw_img = Image.open(raw_path).convert("RGB")
        ground_truth_img = Image.open(gt_path).convert("L")
        raw_img = transform_common(raw_img)
        ground_truth_img = self.gt_to_classes(np.array(ground_truth_img))
        ground_truth_img = transform_common(ground_truth_img)

        #transformed = self.transform_img(image=raw_img, mask=ground_truth_img)
        #raw_img = transformed["image"]
        #ground_truth_img = transformed["mask"]
        #raw_img = self.transform_common(raw_img)
        raw_img = transform_img(raw_img)
        #ground_truth_img = self.transform_common(ground_truth_img)
        
        return raw_img, ground_truth_img

    def gt_to_classes(self, gt: np.ndarray) -> torch.Tensor:
        # Input: (H, W), values: 0-num_classes
        # Output: (num_classes, H, W)
        output = torch.zeros((self.num_classes, *gt.shape))
        for label in labels:
            output[label.id] = torch.Tensor(gt == label.id)
        return output
    
    def classes_to_rgb(self, output: torch.Tensor) -> torch.Tensor:
        # Input: (num_classes, H, W)
        # Output: (3, H, W)
        rgb = torch.zeros((3, *output.size()[-2:]))
        output_max = torch.argmax(output.squeeze(), dim=0)
        for label in labels:
            for c in range(3):
                rgb[c][output_max == label.id] = label.color[c] / 255
        return rgb


class CachedDataset(Dataset):
    def __init__(self):
        pass

Label = namedtuple( "Label" , [
    "name"        , # The identifier of this label, e.g. "car", "person", ... .
                    # We use them to uniquely name a class

    "id"          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    "trainId"     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # "preparation" folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    "category"    , # The name of the category that this label belongs to

    "categoryId"  , # The ID of this category. Used to create ground truth images
                    # on category level.

    "hasInstances", # Whether this label distinguishes between single instances or not

    "ignoreInEval", # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    "color"       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  "unlabeled"            ,  0 ,      255 , "void"            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  "ego vehicle"          ,  1 ,      255 , "void"            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  "rectification border" ,  2 ,      255 , "void"            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  "out of roi"           ,  3 ,      255 , "void"            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  "static"               ,  4 ,      255 , "void"            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  "dynamic"              ,  5 ,      255 , "void"            , 0       , False        , True         , (111, 74,  0) ),
    Label(  "ground"               ,  6 ,      255 , "void"            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  "road"                 ,  7 ,        0 , "flat"            , 1       , False        , False        , (128, 64,128) ),
    Label(  "sidewalk"             ,  8 ,        1 , "flat"            , 1       , False        , False        , (244, 35,232) ),
    Label(  "parking"              ,  9 ,      255 , "flat"            , 1       , False        , True         , (250,170,160) ),
    Label(  "rail track"           , 10 ,      255 , "flat"            , 1       , False        , True         , (230,150,140) ),
    Label(  "building"             , 11 ,        2 , "construction"    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  "wall"                 , 12 ,        3 , "construction"    , 2       , False        , False        , (102,102,156) ),
    Label(  "fence"                , 13 ,        4 , "construction"    , 2       , False        , False        , (190,153,153) ),
    Label(  "guard rail"           , 14 ,      255 , "construction"    , 2       , False        , True         , (180,165,180) ),
    Label(  "bridge"               , 15 ,      255 , "construction"    , 2       , False        , True         , (150,100,100) ),
    Label(  "tunnel"               , 16 ,      255 , "construction"    , 2       , False        , True         , (150,120, 90) ),
    Label(  "pole"                 , 17 ,        5 , "object"          , 3       , False        , False        , (153,153,153) ),
    Label(  "polegroup"            , 18 ,      255 , "object"          , 3       , False        , True         , (153,153,153) ),
    Label(  "traffic light"        , 19 ,        6 , "object"          , 3       , False        , False        , (250,170, 30) ),
    Label(  "traffic sign"         , 20 ,        7 , "object"          , 3       , False        , False        , (220,220,  0) ),
    Label(  "vegetation"           , 21 ,        8 , "nature"          , 4       , False        , False        , (107,142, 35) ),
    Label(  "terrain"              , 22 ,        9 , "nature"          , 4       , False        , False        , (152,251,152) ),
    Label(  "sky"                  , 23 ,       10 , "sky"             , 5       , False        , False        , ( 70,130,180) ),
    Label(  "person"               , 24 ,       11 , "human"           , 6       , True         , False        , (220, 20, 60) ),
    Label(  "rider"                , 25 ,       12 , "human"           , 6       , True         , False        , (255,  0,  0) ),
    Label(  "car"                  , 26 ,       13 , "vehicle"         , 7       , True         , False        , (  0,  0,142) ),
    Label(  "truck"                , 27 ,       14 , "vehicle"         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  "bus"                  , 28 ,       15 , "vehicle"         , 7       , True         , False        , (  0, 60,100) ),
    Label(  "caravan"              , 29 ,      255 , "vehicle"         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  "trailer"              , 30 ,      255 , "vehicle"         , 7       , True         , True         , (  0,  0,110) ),
    Label(  "train"                , 31 ,       16 , "vehicle"         , 7       , True         , False        , (  0, 80,100) ),
    Label(  "motorcycle"           , 32 ,       17 , "vehicle"         , 7       , True         , False        , (  0,  0,230) ),
    Label(  "bicycle"              , 33 ,       18 , "vehicle"         , 7       , True         , False        , (119, 11, 32) ),
    #Label(  "license plate"        , -1 ,       -1 , "vehicle"         , 7       , False        , True         , (  0,  0,142) ),
]

GroupedLabel = namedtuple("GroupedLabel", [
    "id", # group id
    "name", # group name
    "ids", # list of ids
    "color", # color of the group
])
grouped_labels = [
    Label(0, "motor vehicles" , [26, 27, 28, 28, 29, 30, 31, 32],     (  0,   0, 142)),
    Label(1, "pedestrians"    , [24, 25, 33],                         (220,  20,  60)),
    Label(2, "road"           , [6, 7, 8, 9, 10],                     (128,  64, 128)),
    Label(3, "traffic objects", [17, 18, 19, 20],                     (250, 170,  30)),
    Label(4, "background"     , [11, 12, 13, 14, 15, 16, 21, 22, 23], ( 70,  70,  70)),
    Label(5, "void"           , [0, 2, 3, 4, 5],                      (  0,   0,   0)),
    Label(6, "ego vehicle"    , [1],                                  (  0,   0,   0)),
]