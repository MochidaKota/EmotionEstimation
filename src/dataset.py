import torch
from torch.utils.data import Dataset

import pandas as pd
import cv2
from PIL import Image

from utils import util

class ImageList(Dataset):
    def __init__(self, labels_path, video_name_list, au_transform=None, gaze_transform=None):
        self.img_paths = []
        #! format of DataFrame must be same as 'emo_and_au(video1-25).csv' 
        self.labels = pd.read_csv(labels_path)
        self.au_transform = au_transform
        self.gaze_transform = gaze_transform
        
        if len(video_name_list) == 1:
            video_name = video_name_list.item()
            self.img_paths = self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values
        else:   
            for video_name in video_name_list:
                self.img_paths.extend(self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values)
    
        self.img_paths = sorted(self.img_paths)
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        
        au_img = Image.open(img_path)
        if self.au_transform is not None:
            au_img = self.au_transform(au_img)
        
        gaze_img = cv2.imread(img_path)
        if self.gaze_transform is not None:
            gaze_img = self.gaze_transform(gaze_img)
            
        emo = self.labels[self.labels['img_path'] == img_path]['emotion'].values[0]
        emo = torch.tensor(emo, dtype=torch.long)
        
        au = self.labels[self.labels['img_path'] == img_path].iloc[:, 2:].values[0]
        au = torch.tensor(au, dtype=torch.float)
        
        return au_img, gaze_img, img_path, emo, au
    
    def __len__(self):
        return len(self.img_paths)
    
class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        
    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)
    
    def __len__(self):
        return min(len(d) for d in self.datasets)
    
class FeatList(Dataset):
    def __init__(self, labels_path, video_name_list, feats_path):
        self.img_paths = []
        #! format of DataFrame must be same as 'emo_and_au(video1-25).csv'
        self.labels = pd.read_csv(labels_path)
        #! format of DataFrame must be same as 'JAANet_feature.pkl'
        self.feats = pd.read_pickle(feats_path)
        
        if len(video_name_list) == 1:
            video_name = video_name_list[0]
            self.img_paths = self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values
        else:   
            for video_name in video_name_list:
                self.img_paths.extend(self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values)
        
        self.img_paths = sorted(self.img_paths)
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        
        feat = self.feats[self.feats['img_path'] == img_path].iloc[:, 1:].values[0]
        feat = torch.tensor(feat, dtype=torch.float)
        
        emo = self.labels[self.labels['img_path'] == img_path]['emotion'].values[0]
        emo = torch.tensor(emo, dtype=torch.long)
        
        return feat, img_path, emo
    
    def __len__(self):
        return len(self.img_paths)

class SeqFeatList(Dataset):
    def __init__(self, labels_path, video_name_list, feats_path, window_size=30):
        self.img_paths = []
        #! format of DataFrame must be same as 'emo_and_au(video1-25).csv'
        self.labels = pd.read_csv(labels_path)
        #! format of DataFrame must be same as 'JAANet_feature.pkl'
        self.feats = pd.read_pickle(feats_path)
        self.window_size = window_size
        
        if len(video_name_list) == 1:
            video_name = video_name_list[0]
            self.img_paths = self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values
        else:   
            for video_name in video_name_list:
                self.img_paths.extend(self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values)
    
        self.img_paths = sorted(self.img_paths)
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_path_list = util.get_sequence_img_path(img_path, window_size=self.window_size, current_position='head')

        # filterig DataFrame in advance
        filtered_feats = self.feats[self.feats['img_path'].isin(img_path_list)]
        filtered_feats.set_index('img_path', inplace=True)

        # convert to tensor
        tensor_list = []
        for _img_path in img_path_list:
            feat = filtered_feats.loc[_img_path].values
            tensor_list.append(torch.tensor(feat, dtype=torch.float).unsqueeze(0))

        # concat tensor
        feats = torch.cat(tensor_list, dim=0)

        emo = self.labels[self.labels['img_path'] == img_path]['emotion'].values[0]
        emo = torch.tensor(emo, dtype=torch.long)

        return feats, img_path, emo

    def __len__(self):
        return len(self.img_paths)
    
class SeqFeatList2(Dataset):
    #* return emotion_list
    def __init__(self, labels_path, video_name_list, feats_path, window_size=30):
        self.img_paths = []
        #! format of DataFrame must be same as 'emo_and_au(video1-25).csv'
        self.labels = pd.read_csv(labels_path, converters={'emotion_list': eval})
        #! format of DataFrame must be same as 'JAANet_feature.pkl'
        self.feats = pd.read_pickle(feats_path)
        self.window_size = window_size
        
        if len(video_name_list) == 1:
            video_name = video_name_list[0]
            self.img_paths = self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values
        else:   
            for video_name in video_name_list:
                self.img_paths.extend(self.labels[self.labels['img_path'].str.contains(video_name + '/')]['img_path'].values)
    
        self.img_paths = sorted(self.img_paths)
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_path_list = util.get_sequence_img_path(img_path, window_size=self.window_size, current_position='head')

        # filterig DataFrame in advance
        filtered_feats = self.feats[self.feats['img_path'].isin(img_path_list)]
        filtered_feats.set_index('img_path', inplace=True)

        # convert to tensor
        tensor_list = []
        for _img_path in img_path_list:
            feat = filtered_feats.loc[_img_path].values
            tensor_list.append(torch.tensor(feat, dtype=torch.float).unsqueeze(0))

        # concat tensor
        feats = torch.cat(tensor_list, dim=0)

        emo_list = self.labels[self.labels['img_path'] == img_path]['emotion_list'].values[0]
        emo_list = torch.tensor(emo_list, dtype=torch.float)

        return feats, img_path, emo_list

    def __len__(self):
        return len(self.img_paths)