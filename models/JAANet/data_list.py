import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from pathlib import Path
import os
import cv2


def make_dataset(img_paths, emotion, au, nose_landmark):
    len_ = len(img_paths)
    images = [(img_paths[i], emotion[i], au[i], nose_landmark[i, :]) for i in range(len_)]

    return images

def make_dataset_test(img_paths, emotion, nose_landmark):
    len_ = len(img_paths)
    images = [(img_paths[i], emotion[i, :], nose_landmark[i, :]) for i in range(len_)]

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Image value: [0,1]
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)

def _get_img_paths(img_dir):
    img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    img_paths = [p for p in sorted(Path(img_dir).iterdir()) if p.suffix in img_extensions]

    return img_paths

class ImageList(Dataset):
    def __init__(self, crop_size, img_dir, transform=None):
        # 画像ファイルのパス一覧を取得する。
        img_paths = _get_img_paths(img_dir)
        if len(img_paths) == 0:
            raise (RuntimeError('Found 0 images in subfolders of: ' + img_dir + '\n'))
        emotion = np.loadtxt(img_dir + 'emotion.txt')
        au = np.loadtxt(img_dir + 'au.txt')
        nose_landmark = np.loadtxt(img_dir + 'nose_landmark.txt')
        imgs = make_dataset(img_paths, emotion, au, nose_landmark)
        self.imgs = imgs
        self.transform = transform
        self.crop_size = crop_size

    def __getitem__(self, index):
        # インデックスに対応するファイルパスを取得する。
        path , emotion, au, nose_landmark = self.imgs[index]

        # 画像を読み込む。
        img = Image.open(path)
        img = F.resize(img=img, size=(180, 320))
        
        w, h = img.size
        #offset_y = random.randint(0, h - self.crop_size)
        #offset_x = random.randint(0, w - self.crop_size)
            
        center_x = w*nose_landmark[0]
        center_y = h*nose_landmark[1]
            
        # flip = random.randint(0, 1)

        if self.transform is not None:
            img = self.transform(img, center_x, center_y)
            
        return img, emotion, au

    def __len__(self):
        # ディレクトリ内の画像枚数を返す。
        return len(self.imgs)
    
emo_label_list = {
    'others': 0,
    'comfort': 1,
    'discomfort': 2
}

class ImageListV2(Dataset):
    def __init__(self, img_dir, label_dir, mode, transform=None):
        self.img_paths = []
        self.transform = transform
        '''csv type must be same as emo_and_au(video1-25).csv'''
        self.labels = pd.read_csv(label_dir)
        
        video_name_list = np.loadtxt("/mnt/iot-qnap3/mochida/medical-care/EmoEstimateByJAANet/data_v2/" + mode + '/video_name_list.txt', dtype=str)
        if video_name_list.ndim == 0:
            video_name = video_name_list.item()
            self.img_paths = self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values
        else:   
            for video_name in video_name_list:
                self.img_paths.extend(self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values)
    
        self.img_paths = sorted(self.img_paths)

        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        path = img_path.replace('../', '/mnt/iot-qnap3/mochida/medical-care/EmoEstimateByJAANet/')
        
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        
        emotion = self.labels[self.labels["img_path"] == img_path].emotion.values[0]
        emotion = torch.tensor(emotion, dtype=torch.long)
        
        au = self.labels[self.labels["img_path"] == img_path].iloc[:, 2:].values[0]
        au = torch.tensor(au, dtype=torch.float)
        
        return img, path, emotion, au
    
    def __len__(self):
        return len(self.img_paths)
    
    def get_num_per_class(self, type=None):
        num_per_class = []
        labels = self.labels[self.labels['img_path'].isin(self.img_paths)]
        
        if type in emo_label_list.keys():
            num_per_class.append(len(labels[(~labels["emotion"].isin([emo_label_list[type]]))]))
            num_per_class.append(len(labels[(labels["emotion"].isin([emo_label_list[type]]))]))
        
        else:
            for i in range(self.labels["emotion"].unique().shape[0]):
                num_per_class.append(len(labels[(labels["emotion"].isin([i]))]))
    
        return num_per_class
    
class ImageListV3(Dataset):
    def __init__(self, label_dir, mode, au_transform=None, gaze_feat_dir=None):
        self.img_paths = []
        self.au_transform = au_transform
        self.gaze_midfeat = pd.read_csv(gaze_feat_dir)
        '''csv type must be same as emo_and_au(video1-25).csv'''
        self.labels = pd.read_csv(label_dir)
        
        video_name_list = np.loadtxt("/mnt/iot-qnap3/mochida/medical-care/EmoEstimateByJAANet/data_v2/" + mode + '/video_name_list.txt', dtype=str)
        if video_name_list.ndim == 0:
            video_name = video_name_list.item()
            self.img_paths = self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values
        else:   
            for video_name in video_name_list:
                self.img_paths.extend(self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values)
    
        self.img_paths = sorted(self.img_paths)

        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        path = img_path.replace('../', '/mnt/iot-qnap3/mochida/medical-care/EmoEstimateByJAANet/')
        
        au_img = Image.open(path)
        if self.au_transform is not None:
            au_img = self.au_transform(au_img)
            
        # gaze_img = cv2.imread(path)
        # if self.gaze_transform is not None:
        #     gaze_img = self.gaze_transform(gaze_img)
        
        gaze_feat = self.gaze_midfeat[self.gaze_midfeat["img_path"] == path].iloc[:, 1:].values[0]
        gaze_feat = torch.tensor(gaze_feat, dtype=torch.float)
        
        emotion = self.labels[self.labels["img_path"] == img_path].emotion.values[0]
        emotion = torch.tensor(emotion, dtype=torch.long)
        
        au = self.labels[self.labels["img_path"] == img_path].iloc[:, 2:].values[0]
        au = torch.tensor(au, dtype=torch.float)
        
        return au_img, gaze_feat, path, emotion, au
    
    def __len__(self):
        return len(self.img_paths)
    
    def get_num_per_class(self, type=None):
        num_per_class = []
        labels = self.labels[self.labels['img_path'].isin(self.img_paths)]
        
        if type in emo_label_list.keys():
            num_per_class.append(len(labels[(~labels["emotion"].isin([emo_label_list[type]]))]))
            num_per_class.append(len(labels[(labels["emotion"].isin([emo_label_list[type]]))]))
        
        else:
            for i in range(self.labels["emotion"].unique().shape[0]):
                num_per_class.append(len(labels[(labels["emotion"].isin([i]))]))
    
        return num_per_class
    
class ImageListV4(Dataset):
    def __init__(self, label_dir, mode, au_transform=None, gaze_transform=None):
        self.img_paths = []
        self.au_transform = au_transform
        self.gaze_transform = gaze_transform
        '''csv type must be same as emo_and_au(video1-25).csv'''
        self.labels = pd.read_csv(label_dir)
        
        video_name_list = np.loadtxt("/mnt/iot-qnap3/mochida/medical-care/EmoEstimateByJAANet/data_v2/" + mode + '/video_name_list.txt', dtype=str)
        if video_name_list.ndim == 0:
            video_name = video_name_list.item()
            self.img_paths = self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values
        else:   
            for video_name in video_name_list:
                self.img_paths.extend(self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values)
    
        self.img_paths = sorted(self.img_paths)

        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        path = img_path.replace('../', '/mnt/iot-qnap3/mochida/medical-care/EmoEstimateByJAANet/')
        
        au_img = Image.open(path)
        if self.au_transform is not None:
            au_img = self.au_transform(au_img)
            
        gaze_img = cv2.imread(path)
        if self.gaze_transform is not None:
            gaze_img = self.gaze_transform(gaze_img)
        
        emotion = self.labels[self.labels["img_path"] == img_path].emotion.values[0]
        emotion = torch.tensor(emotion, dtype=torch.long)
        
        au = self.labels[self.labels["img_path"] == img_path].iloc[:, 2:].values[0]
        au = torch.tensor(au, dtype=torch.float)
        
        return au_img, gaze_img, path, emotion, au
    
    def __len__(self):
        return len(self.img_paths)
    
    def get_num_per_class(self, type=None):
        num_per_class = []
        labels = self.labels[self.labels['img_path'].isin(self.img_paths)]
        
        if type in emo_label_list.keys():
            num_per_class.append(len(labels[(~labels["emotion"].isin([emo_label_list[type]]))]))
            num_per_class.append(len(labels[(labels["emotion"].isin([emo_label_list[type]]))]))
        
        else:
            for i in range(self.labels["emotion"].unique().shape[0]):
                num_per_class.append(len(labels[(labels["emotion"].isin([i]))]))
    
        return num_per_class
    
class MidFeatList(Dataset):
    def __init__(self, label_dir, mode, au_midfeat_dir=None, gaze_midfeat_dir=None):
        self.img_paths = []
        '''csv type must be same as emo_and_au(video1-25).csv'''
        self.labels = pd.read_csv(label_dir)
        self.au_midfeat = pd.read_csv(au_midfeat_dir)
        self.gaze_midfeat = pd.read_csv(gaze_midfeat_dir)
        
        video_name_list = np.loadtxt("/mnt/iot-qnap3/mochida/medical-care/EmoEstimateByJAANet/data_v2/" + mode + '/video_name_list.txt', dtype=str)
        if video_name_list.ndim == 0:
            video_name = video_name_list.item()
            self.img_paths = self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values
        else:   
            for video_name in video_name_list:
                self.img_paths.extend(self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values)
    
        self.img_paths = sorted(self.img_paths)

        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        path = img_path.replace('../', '/mnt/iot-qnap3/mochida/medical-care/EmoEstimateByJAANet/')
        
        au_midfeat = self.au_midfeat[self.au_midfeat["img_path"] == path].iloc[:, 1:].values[0]
        au_midfeat = torch.tensor(au_midfeat, dtype=torch.float)
        
        gaze_midfeat = self.gaze_midfeat[self.gaze_midfeat["img_path"] == path].iloc[:, 1:].values[0]
        gaze_midfeat = torch.tensor(gaze_midfeat, dtype=torch.float)
        
        emotion = self.labels[self.labels["img_path"] == img_path].emotion.values[0]
        emotion = torch.tensor(emotion, dtype=torch.long)
        
        au = self.labels[self.labels["img_path"] == img_path].iloc[:, 2:].values[0]
        au = torch.tensor(au, dtype=torch.float)
        
        return au_midfeat, gaze_midfeat, path, emotion, au
    
    def __len__(self):
        return len(self.img_paths)
    
    def get_num_per_class(self, type=None):
        num_per_class = []
        labels = self.labels[self.labels['img_path'].isin(self.img_paths)]
        
        if type in emo_label_list.keys():
            num_per_class.append(len(labels[(~labels["emotion"].isin([emo_label_list[type]]))]))
            num_per_class.append(len(labels[(labels["emotion"].isin([emo_label_list[type]]))]))
        
        else:
            for i in range(self.labels["emotion"].unique().shape[0]):
                num_per_class.append(len(labels[(labels["emotion"].isin([i]))]))
    
        return num_per_class
    
class Sequence_ImageList(Dataset):
    def __init__(self, length, label_dir, mode, transform=None):
        self.img_paths = []
        self.transform = transform
        '''csv type must be same as emo_and_au(video1-25).csv'''
        self.labels = pd.read_csv(label_dir)

        video_name_list = np.loadtxt("/mnt/iot-qnap3/mochida/medical-care/EmoEstimateByJAANet/data_v2/" + mode + '/video_name_list.txt', dtype=str)
        for video_name in video_name_list:
            self.img_paths.extend(self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values)
        # self.img_paths = sorted(self.img_paths)
            
        self.length = length
        self.channel = 3
        self.width = 176
        self.height = 176
        self.seq_img_paths = []
        # make sequence image path list
        # ex) [1,2,3,4,5,6,7,8,9,10] -> [[1,2,3,4,5], [6,7,8,9,10]]
        while len(self.img_paths) > self.length:
            self.seq_img_paths.append(self.img_paths[:self.length])
            self.img_paths = self.img_paths[self.length:]
        
    def __getitem__(self, index):
        seq_img_path = self.seq_img_paths[index]
        seq_img = torch.empty((self.length, self.channel, self.width, self.height))
        seq_emo_list = []
        seq_au_list = []
        for i, img_path in enumerate(seq_img_path):
            img = Image.open("../" + img_path)
            if self.transform is not None:
                img = self.transform(img)
            seq_img[i] = img
            
            emotion = self.labels[self.labels["img_path"] == img_path].emotion.values[0]
            seq_emo_list.append(emotion)
            
            au = self.labels[self.labels["img_path"] == img_path].iloc[:, 2:].values[0]
            seq_au_list.append(au)
        
        # Returns 1 if there is at least one 1 in seq_emo_list
        # Returns 2 if there is at least one 2 in seq_emo_list
        # Returns 0 if there is no 1 or 2 in seq_emo_list
        if 1 in seq_emo_list:
            seq_emotion = torch.tensor(1, dtype=torch.long)
        elif 2 in seq_emo_list:
            seq_emotion = torch.tensor(2, dtype=torch.long)
        else:
            seq_emotion = torch.tensor(0, dtype=torch.long)
        
        return seq_img, seq_img_path, seq_emotion, seq_au_list
    
    def __len__(self):
        return len(self.seq_img_paths)
        