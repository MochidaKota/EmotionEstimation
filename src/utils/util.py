import torch
import numpy as np
import pandas as pd
import random
import os

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    
def get_video_name_list(video_name_list_path, fold=0, phase='train'):
    video_name_list = pd.read_csv(video_name_list_path)
    
    if fold == 0:
        video_name_list = video_name_list.columns.values.tolist()
    else:
        if phase == 'train':
            video_name_list = video_name_list.columns[video_name_list.iloc[fold - 1] == 0].values.tolist()
        elif phase == 'test':
            video_name_list = video_name_list.columns[video_name_list.iloc[fold - 1] == 1].values.tolist()
        
    return video_name_list

def str2bool(v):
    return v.lower() in ('true')

def get_video_name_and_frame_num(img_path):
    basename = os.path.splitext(os.path.basename(img_path))[0]
    video_name, frame_num = basename.split('_')
    frame_num = int(frame_num)
    return video_name, frame_num

def get_sequence_img_path(current_img_path, window_size=30, current_position='tail'):
    img_path_list = []
    root_dir = os.path.dirname(current_img_path)
    
    video_name, current_frame = get_video_name_and_frame_num(current_img_path)
    
    if current_position == 'tail':
        for i in range(0, window_size):
            past_frame = current_frame - (window_size - i - 1)
            if past_frame < 0:
                img_path_list.append('/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/interim/black.jpg')
            else:
                img_path_list.append(os.path.join(root_dir, f'{video_name}_{str(past_frame).zfill(4)}.jpg'))
    
    elif current_position == 'head':
        for i in range(0, window_size):
            new_frame = current_frame + i
            img_path = os.path.join(root_dir, f'{video_name}_{str(new_frame).zfill(4)}.jpg')
            if os.path.exists(img_path) == False:
                img_path_list.append('/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/interim/black.jpg')
            else:
                img_path_list.append(img_path)
            
    return img_path_list

def get_diff_img_path(current_img_path, distance):
    img_path_list = []
    root_dir = os.path.dirname(current_img_path)
    
    video_name, current_frame = get_video_name_and_frame_num(current_img_path)
    
    past_frame = current_frame - distance
    
    if past_frame < 0:
        img_path_list.append('/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/interim/black.jpg')
    else:
        img_path_list.append(os.path.join(root_dir, f'{video_name}_{str(past_frame).zfill(4)}.jpg'))
        
    img_path_list.append(current_img_path)
    
    return img_path_list

def get_sequence_list(video_name, window_size=30, shift_size=15, threshold=0.5, drop_mixed=False, label_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/emo_and_au-gaze-hp(video1-25).csv'):
    label_df = pd.read_csv(label_path)
    label_df['video_name'], _ = zip(*label_df['img_path'].map(get_video_name_and_frame_num))
    
    sequence_list = pd.DataFrame(columns=['img_path', 'emotion', 'positive_rate', 'negative_rate'])
    head_img_paths = []
    emotions = []
    positive_rate = []
    negative_rate = []
    
    video_df = label_df[label_df['video_name'] == video_name]
    video_df = video_df.reset_index(drop=True)
    
    point = 0
    
    while point + window_size <= len(video_df):
        head_img_paths.append(video_df['img_path'][point])
        _emotions = video_df['emotion'][point:point+window_size].tolist()
        unique_values = list(set(_emotions))
        
        if len(unique_values) == 1:
            emotions.append(unique_values[0])
            if unique_values[0] == 0:
                negative_rate.append(1.0)
                positive_rate.append(0.0)
            else:
                negative_rate.append(0.0)
                positive_rate.append(1.0)
        elif len(unique_values) == 2:
            # more than threshold
            if _emotions.count(unique_values[1]) / window_size >= threshold:
                emotions.append(unique_values[1])
            else:
                emotions.append(unique_values[0])
            
            negative_rate.append(_emotions.count(unique_values[0]) / window_size)
            positive_rate.append(_emotions.count(unique_values[1]) / window_size)
            
        point += shift_size
    
    sequence_list['img_path'] = head_img_paths
    sequence_list['emotion'] = emotions
    sequence_list['positive_rate'] = positive_rate
    sequence_list['negative_rate'] = negative_rate
    
    if drop_mixed:
        sequence_list = sequence_list[(sequence_list['positive_rate'] == 1.0) | (sequence_list['negative_rate'] == 1.0)]
        sequence_list = sequence_list.reset_index(drop=True)
    
    return sequence_list

def normarize_gaze_and_hp(input, is_delta=False):
    # pitch, yawの範囲を[-90,90]だと仮定し，-1~1に正規化
    if is_delta:
        input = input / 180
    else:
        input = input / 90
    
    # -1~1の範囲を超える値を-1~1にクリップ
    if type(input) == pd.core.series.Series:
        input = input.apply(lambda x: 1 if x > 1 else x)
        input = input.apply(lambda x: -1 if x < -1 else x)
    else:
        input = np.clip(input, -1, 1)
        
    # 0~1に正規化
    normed_input = (input + 1) / 2
    return normed_input

def standardize_feature(feat_list):
    for i in range(len(feat_list)):
        mean = torch.mean(feat_list[i], dim=1, keepdim=True)
        std = torch.std(feat_list[i], dim=1, keepdim=True)
        feat_list[i] = (feat_list[i] - mean) / std
        
    return feat_list

def convert_label_to_binary(labels, target_emo):
    '''
    Args:
        labels: tensor
        target_emo: str 'comfort' or 'discomfort'
    '''
    if target_emo == 'comfort':
        labels = torch.where(labels == 2, torch.tensor(0), labels)
    elif target_emo == 'discomfort':
        labels = torch.where(labels == 1, torch.tensor(0), labels)
        labels = torch.where(labels == 2, torch.tensor(1), labels)
        
    return labels