import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import os
import random
import numpy as np
from skimage import io
from skimage import color
from skimage.transform import rotate, resize
from PIL import Image
from torchvision import utils as vutils
import pandas as pd
from pandas import DataFrame
import sys
import pdb
import pickle
from scipy import *

# sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append('/cvlabdata2/home/ziyi/6D-Pose/ZZNET')
from utilis import *
# from .. import utilis

class ZZNETDataset(Dataset):
    def __init__(self, raw_image=False, grayscale=False, augment=False, batch=True, root_dir = '/cvlabdata2/home/ziyi/6D-Pose/dataset/train/train'):
        self.preprocessing(raw_image=raw_image, grayscale=grayscale, batch=batch, augment=augment, root_dir=root_dir)
   
    def preprocessing(self, root_dir, grayscale, augment, raw_image, batch, img_h=480):
        """ 
        Args:
            root_dir: dir for the train_dataset
            grayscale: Rgb2Gray 
            augment: Use random data augmentation
            raw_img: Return raw RGB image w/o any augmentation or normalization for post-processing
            img_height: RGB images are rescaled to this maximum height
        """
        self.batch = batch
        self.grayscale = grayscale
        self.augment = augment
        self.raw_image = raw_image
        self.image_height = img_h

        # Return raw RGB images w/o augmentation or normalization
        # Warning: this option superposes the other parameters, and should be used w/ care.
        if self.raw_image:
            self.augment = False
            self.grayscale = False
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.image_height),
                transforms.ToTensor()
            ])
        else:
            if self.grayscale:
                self.image_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(self.image_height),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        # mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                        # std=[0.25]
                        # urbanscape statistics (should generalize well enough)
                        mean=[0.4308],
                        std=[0.1724]
                        # naturescape statistics (backup)
                        # mean=[0.4084],
                        # std=[0.1404]
                    )
                ])
            else:
                self.image_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(self.image_height),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        # urbanscape statistics (should generalize well enough)
                        mean=[0.4245, 0.4375, 0.3836],
                        std=[0.1823, 0.1701, 0.1854]
                        # naturescape statistics (backup)
                        # mean = [0.3636, 0.4331, 0.2956],
                        # std = [0.1383, 0.1457, 0.1147]
                    )
                ])
            
        self.rgb_files = []
        if isinstance(root_dir, list):
            root_dir_ls = root_dir
        elif os.path.isdir(root_dir):
            root_dir_ls = [root_dir]
        else:
            raise ValueError("root_dir type {} is not supported!".format(type(root_dir)))
       
        for base_dir in root_dir_ls:
            rgb_dir = base_dir + '/rgb/'
         
        _rgb_files = os.listdir(rgb_dir)
        _rgb_files = [rgb_dir + f for f in _rgb_files]
        _rgb_files.sort()
        self.rgb_files.extend(_rgb_files)  # list len=15000

        with open('/cvlabdata2/home/ziyi/6D-Pose/ZZNET/Data/GT_tuple_loca_ang.pickle', "rb") as f:
            data_tuple = pickle.load(f)
        self.pos_pool = data_tuple['data'][1] # torch.Size([15000, 32])
        self.neg_pool = data_tuple['data'][2] # torch.Size([15000, 32])
        # pdb.set_trace()
        
        """ !!!important""" 
        #for test make model  faster
        self.pos_pool = data_tuple['data'][1][0:160,:] # torch.Size([15000, 32])
        self.neg_pool = data_tuple['data'][2][0:160,:] # torch.Size([15000, 32])
        self.rgb_files_2=self.rgb_files[0:160]

    def __getitem__(self, index):
        """
        Args
            index:0-len(dataset), int

        Return
            q_image: tensor (3/1*480*720)<---(channel_rgb/grey, height, width)
            pos_img: same
            neg_img: same
        """
        aug_rotation=30
        aug_scale_min=2 / 3
        aug_scale_max=3 / 2
        aug_contrast=0.1
        aug_brightness=0.1
        
        # q_img
        # image = io.imread(pos_img_id) AttributeError: module 'scipy.io' has no attribute 'imread'?
        image = np.array(Image.open(self.rgb_files[index]).convert('RGB')) # modify the image path according to your need
        
        # pos_img, neg_img
        pos_imgs = self.pos_pool[index] #torch.Size([32])
        neg_imgs = self.neg_pool[index] #torch.Size([32])
        # id_pos_neg = torch.LongTensor(random.sample(range(pos_imgs.shape[0]), 1))
        # pos_img_id = torch.index_select(pos_imgs, 0, id_pos_neg).item()
        # neg_img_id = torch.index_select(neg_imgs, 0, id_pos_neg).item()
        pos_img_id = pos_imgs[-1]
        neg_img_id = neg_imgs[0]
        pos_img = np.array(Image.open(self.rgb_files[pos_img_id]).convert('RGB'))
        neg_img = np.array(Image.open(self.rgb_files[neg_img_id]).convert('RGB'))

        if len(image.shape) < 3:
            image = color.gray2rgb(image)
            pos_img = color.gray2rgb(pos_img)
            neg_img = color.gray2rgb(neg_img)
        if len(image.shape) == 3 and image.shape[-1] == 4:
            # RGBA to RGB for Cesium dataset
            image = image[:, :, :3]
            pos_img = pos_img[:, :, :3]
            neg_img = neg_img[:, :, :3]

        if self.augment:
            # if mini-batch size is larger than 1, resizing is done in the collate_fn after the batch data is fetched.
            if self.batch:
                scale_factor = 1.0
                angle = 0.0
            else:
                scale_factor = random.uniform(aug_scale_min, aug_scale_max)
                angle = random.uniform(-aug_rotation, aug_rotation)
            
            # augment input image
            if self.grayscale:
                cur_image_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(int(self.image_height * scale_factor)),
                    transforms.Grayscale(),
                    transforms.ColorJitter(brightness=aug_brightness, contrast=aug_contrast),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        # EPFL statistics (should generalize well enough)
                        mean=[0.4670],
                        std=[0.1998]
                        # comballaz statistics
                        # mean=[0.3831],
                        # std=[0.1148]
                    )
                ])
            else:
                cur_image_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(int(self.image_height * scale_factor)),
                    transforms.ColorJitter(brightness=aug_brightness, contrast=aug_contrast),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        # EPFL statistics (should generalize well enough)
                        mean=[0.4641, 0.4781, 0.3662],
                        std=[0.2123, 0.1976, 0.2189]
                        # comballaz statistics
                        # mean = [0.3311, 0.4055, 0.3130],
                        # std = [0.1146, 0.1200, 0.0903]
                    )
                ])
            self.image = cur_image_transform(image)
            self.pos_img = cur_image_transform(pos_img)
            self.neg_img = cur_image_transform(neg_img)

            # rotate input image
            def my_rot(t, angle, order, mode='constant'):
                t = t.permute(1, 2, 0).numpy()
                t = rotate(t, angle, order=order, mode=mode, cval=-1)
                t = torch.from_numpy(t).permute(2, 0, 1).float()
                return t

            self.image = my_rot(self.image, angle, 1, 'constant')
            self.pos_img = my_rot(self.pos_img, angle, 1, 'constant')
            self.neg_img = my_rot(self.neg_img, angle, 1, 'constant')
        else:
            self.image = self.image_transform(image)
            self.pos_img = self.image_transform(pos_img)
            self.neg_img = self.image_transform(neg_img)
       
        
        #save tuple images
        tup_img_path = 'Data/img_tuples_loc+ang/'
        if index%100==0:
            q_img_name = str(index) + '-Q-' + str(self.rgb_files[index])[-23:]
            pos_img_name = str(index) + '-P-' + str(self.rgb_files[pos_img_id])[-23:]
            neg_img_name = str(index) + '-N-' + str(self.rgb_files[neg_img_id])[-23:]
            vutils.save_image(self.image, tup_img_path + q_img_name)
            vutils.save_image(self.pos_img, tup_img_path + pos_img_name)
            vutils.save_image(self.neg_img, tup_img_path + neg_img_name)
    
        #save tuple WGS64 Position[latitude-longitude-height]
        save_tuple_wgs(index=index, pos_img_id=pos_img_id, neg_img_id=neg_img_id, save_path='Data/tuple_loc+ang.csv')
        
        return self.image, self.pos_img, self.neg_img, index, pos_img_id, neg_img_id


    def __len__(self):
        return len(self.rgb_files_2)



        