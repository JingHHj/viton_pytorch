import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import os
import numpy as np
import json
import pdb



class GeneratorDataset(Dataset):
    def __init__(self, root_dir = "./data", image_size=(256,192), transform=None):
        self.root_dir = root_dir
        self.image_size = image_size    
        self.transform = transforms.Compose([
            transforms.Resize(image_size),   # in the paper all the input images are 256*192
            transforms.ToTensor()
        ])

        self.data_dir = os.path.join(self.root_dir, "train")
        self.image_name = []
        self.cloth_name = []
        with open(os.path.join(self.root_dir, "train_pairs.txt"), "r", encoding="utf-8") as f:
            for line in f:
                im, c = line.strip().split()
                self.image_name.append(im)
                self.cloth_name.append(c) 
                
        

    def __len__(self):
        return len(self.image_name)
    

    def _get_dirs(self, index):
        """
            Get the directories of the image, cloth, cloth-mask, pase and keypoint
            Args:
                index: the index of the image
            Return:
                dirs: a dict of the directories of the image, cloth, cloth-mask, pase and keypoint
        """
        cloth_dir = os.path.join(self.data_dir, "cloth", self.cloth_name[index])
        cloth_mask_dir = os.path.join(self.data_dir, "cloth-mask", self.cloth_name[index])
        image_dir = os.path.join(self.data_dir, "image", self.image_name[index])
        segmentation_dir = os.path.join(self.data_dir, "image-parse", self.image_name[index].replace(".jpg", ".png"))
        keypoint_dir = os.path.join(self.data_dir, "pose", self.image_name[index].replace(".jpg", "_keypoints.json"))

        return {"cloth": cloth_dir, 
                "cloth_mask": cloth_mask_dir, 
                "image": image_dir, 
                "segmentation": segmentation_dir, 
                "pose_map": keypoint_dir}
        
    def _load_keypoint(self, keypoint_dir):
        """
            Load the keypoint from the json file into a pose map (torch.Tensor)
            Args:
                keypoint_dir: the directory of the keypoint json file
            Return:
                pose_map: a tensor of the pose map, shape (256, 192, 18)
                
                    - pose_map[:,:,i] only has 1s in the 11*11 square around the i-th keypoint
        """

        # initialize the pose map with everything zero
        pose_map = torch.zeros((18,self.image_size[0], self.image_size[1]))

        # the pose map for visualization
        pose_map_vi = torch.zeros((1,self.image_size[0], self.image_size[1]))

        # open the json file and load the data
        with open(keypoint_dir, "r") as f:
            data = json.load(f)
            data = np.array(data["people"][0]["pose_keypoints"]).reshape(-1, 3)
            # data in the size of (N, 3), N is the number of keypoints

        for idx,(x,y,_) in enumerate(data):
            # data in the form of (x, y, confidence)
            x = np.round(x).astype(int)
            y = np.round(y).astype(int)
            # draw a 11 * 11 square around the keypoint
            pose_map[idx, y-5:y+5, x-5:x+5] = 1
            pose_map_vi[0, y-5:y+5, x-5:x+5] = 1

        return pose_map, pose_map_vi 
        # (256, 192, 18), (1, 256, 192)
    
    def _get_body_representations(self, segmentation, image):
        """
                Get the body representations from the segmentation and image
            Args:
                segmentation: the segmentation tensor, shape (1, 256, 192)
                image: the image tensor, shape (3, 256, 192)
            Return - face_hair: the face and hair part of the image, shape (3, 256, 192)
                    - body_mask: the mask of the body part of the image, shape (1, 256, 192)
        """
        body_parts = segmentation.unique()


        face_hair_mask = (segmentation[0] == body_parts[1]) | \
                         (segmentation[0] == body_parts[4])
        
        body_mask = (
                    (segmentation[0] != body_parts[0]) & \
                    (segmentation[0] !=body_parts[1])  & \
                    (segmentation[0] !=body_parts[4])
                    ).unsqueeze(0)
    
        face_hair = image * face_hair_mask


        return face_hair, body_mask
        

    def __getitem__(self, index):
        """
            Get the item of the dataset
            Args:
                index: the index of the image
            Return:
                info: a dict of the image, cloth, cloth-mask, parse and pose_map
                    - image: the image tensor, shape (3, 256, 192)
                    - cloth: the cloth tensor, shape (3, 256, 192)
                    - cloth-mask: the cloth-mask tensor, shape (1, 256, 192)
                    - segmentation: the parse tensor, shape (1, 256, 192)
                    - pose_map: the keypoint tensor, shape (18, 256, 192)
                    - representation: the alothing-agonistic person representation, shape (22, 256, 192)
        """

        dirs = self._get_dirs(index)
        info = dirs
        info["cloth_name"] = self.cloth_name[index]
        info["image_name"] = self.image_name[index]
        info["cloth"] = self.transform(Image.open(info["cloth"])) # (3, 256, 192)
        info["cloth_mask"] = self.transform(Image.open(info["cloth_mask"])) # (1, 256, 192)
        info["image"] = self.transform(Image.open(info["image"])) # (3, 256, 192)
        info["segmentation"] = self.transform(Image.open(info["segmentation"])) # (1, 256, 192)
        info["pose_map"],info["pose_map_vi"] = self._load_keypoint(info["pose_map"]) # (18, 256, 192)
        # (3, 256, 192), (1, 256, 192) 
        # the second

        info["face_hair_rgb"],info["body_rep"] = self._get_body_representations(info["segmentation"], info["image"])
        info["agonistic_rep"] = torch.cat([info["pose_map"], info["body_rep"], info["face_hair_rgb"]], dim=0)
        # alothing-agonistic person representation:(22,256,192) 
        # 22 = 18+1+3

        return info
        




    

        