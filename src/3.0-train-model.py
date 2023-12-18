"""
* This script is used to train the AU-Gaze-HP model.
* AU:seq_feat, Gaze:seq_feat, HP:seq_feat
* Perform encoding using self-attention with Transformer to consider information changes.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.util import torch_fix_seed, get_video_name_list, str2bool
from dataset import ConcatDataset, SeqFeatList
from networks import EmotionEstimator

from tqdm import tqdm
import json
import pandas as pd

def main(config):
    # fixing seed
    torch_fix_seed()
    
    # setting use_feat_list
    use_feat_list = {'AU':config.use_feat_list[0], 'Gaze':config.use_feat_list[1], 'HP':config.use_feat_list[2]}
    
    # setting dataset and dataloader
    datasets = {}
    dataloaders = {}
    
    datasets_temp = []
    if use_feat_list['AU'] == 1:
        au_dataset = SeqFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='train'),
            feats_path=config.au_feats_path,
            window_size=config.window_size
        )
        datasets_temp.append(au_dataset)
        
    if use_feat_list['Gaze'] == 1:
        gaze_dataset = SeqFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='train'),
            feats_path=config.gaze_feats_path,
            window_size=config.window_size
        )
        datasets_temp.append(gaze_dataset)
        
    if use_feat_list['HP'] == 1:
        hp_dataset = SeqFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='train'),
            feats_path=config.hp_feats_path,
            window_size=config.window_size
        )
        datasets_temp.append(hp_dataset)
    
    datasets['train'] = ConcatDataset(datasets_temp)
    
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    if config.val_phase == True:
        datasets_temp = []
        if use_feat_list['AU'] == 1:
            au_dataset = SeqFeatList(
                labels_path=config.labels_path,
                video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
                feats_path=config.au_feats_path,
                window_size=config.window_size
            )
            datasets_temp.append(au_dataset)
            
        if use_feat_list['Gaze'] == 1:
            gaze_dataset = SeqFeatList(
                labels_path=config.labels_path,
                video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
                feats_path=config.gaze_feats_path,
                window_size=config.window_size
            )
            datasets_temp.append(gaze_dataset)
            
        if use_feat_list['HP'] == 1:
            hp_dataset = SeqFeatList(
                labels_path=config.labels_path,
                video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
                feats_path=config.hp_feats_path,
                window_size=config.window_size
            )
            datasets_temp.append(hp_dataset)
        
        datasets['val'] = ConcatDataset(datasets_temp)
        
        dataloaders['val'] = DataLoader(
            datasets['val'],
            batch_size=1,
            shuffle=False,
            num_workers=2
        )
    
    print()
    print('----- Dataset and Dataloader Setting -----')
    
    # setting networks
    training_module = []
    
    #* Emotion Estimator
    input_dim = 0
    if use_feat_list['AU'] == 1:
        au_input_dim = pd.read_pickle(config.au_feats_path).shape[1] - 1
        input_dim += au_input_dim
    if use_feat_list['Gaze'] == 1:
        gaze_input_dim = pd.read_pickle(config.gaze_feats_path).shape[1] - 1
        input_dim += gaze_input_dim
    if use_feat_list['HP'] == 1:
        hp_input_dim = pd.read_pickle(config.hp_feats_path).shape[1] - 1
        input_dim += hp_input_dim

    emo_net = EmotionEstimator.TransformerClassifier(
        num_classes=config.emo_num,
        input_dim=input_dim,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_hid=config.d_hid,
        num_layers=config.num_layers,
        max_seq_len=config.window_size,
        dropout=config.dropout,
        wo_pe=config.wo_pe,
        pool_type=config.pool_type
    )
        
    # setting device
    device = torch.device("cuda:" + config.gpu_id if torch.cuda.is_available() else "cpu")
    emo_net.to(device)
    
    print()
    print('----- networks loaded -----')
    
    # setting optimizer
    optimizer = None
    if config.optimizer == 'SGD':
        optimizer = optim.SGD(
            params=emo_net.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(
            params=emo_net.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    
    # setting criterion
    criterion = nn.CrossEntropyLoss()
    
    print()
    print(f"----- Training of {config.target_emo} estimator start! (fold{config.fold}) ------")
    if config.val_phase == True:
        phases = ['train', 'val']
        history = {'epoch':[], 'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[]}
    else:
        phases = ['train']
        history = {'epoch':[], 'train_loss':[], 'train_acc':[]}   
        
    for epoch in range(config.n_epochs):
        print('-------------')
        print(f" Epoch{epoch + 1}/{config.n_epochs}")
        history['epoch'].append(epoch+1)
        
        # setting directory for each epoch
        save_path_dir = config.write_path_prefix + config.run_name + f'/epoch{epoch+1}'
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)
        save_each_path_dir = save_path_dir + '/fold' + str(config.fold)
        if not os.path.exists(save_each_path_dir):
            os.makedirs(save_each_path_dir)
        
        save_res_dir = config.write_res_prefix + config.run_name + f'/epoch{epoch+1}'
        if not os.path.exists(save_res_dir):
            os.makedirs(save_res_dir)
        save_each_res_dir = save_res_dir + '/fold' + str(config.fold)
        if not os.path.exists(save_each_res_dir):
            os.makedirs(save_each_res_dir)
            
        # Each epoch has a training and validation phase    
        for phase in phases:
            if phase == 'train':
                emo_net.train()
            else:
                emo_net.eval()
                
            epoch_loss = 0.0  # sum of loss in epoch
            epoch_corrects = 0 # sum of corrects in epoch
                
            for i, batch in tqdm(enumerate(dataloaders[phase])):
                # emo_label_list = {'others': 0, 'comfort': 1, 'discomfort': 2}
                feats_list = []
                count = 0
                if use_feat_list['AU'] == 1:
                    au_seq_feats, _, emotions = batch[count]
                    au_seq_feats = au_seq_feats.to(device)
                    feats_list.append(au_seq_feats)
                    count += 1
                if use_feat_list['Gaze'] == 1:
                    if count == 0:
                        gaze_seq_feats, _, emotions = batch[count]
                    else:
                        gaze_seq_feats, _, _ = batch[count]
                        
                    gaze_seq_feats = gaze_seq_feats.to(device)
                    feats_list.append(gaze_seq_feats)
                    count += 1
                if use_feat_list['HP'] == 1:
                    if count == 0:
                        hp_seq_feats, _, emotions = batch[count]
                    else:
                        hp_seq_feats, _, _ = batch[count]
                    hp_seq_feats = hp_seq_feats.to(device)
                    feats_list.append(hp_seq_feats)
                    count += 1
                    
                _feats = None
                if use_feat_list['AU'] == 1:
                    if _feats is None:
                        _feats = au_seq_feats 
                if use_feat_list['Gaze'] == 1:
                    if _feats is None:
                        _feats = gaze_seq_feats
                if use_feat_list['HP'] == 1:  
                    if _feats is None:
                        _feats = hp_seq_feats
                
                if config.target_emo == 'comfort':
                    emotions = torch.where(emotions == 2, torch.tensor(0), emotions)
                elif config.target_emo == 'discomfort':
                    emotions = torch.where(emotions == 1, torch.tensor(0), emotions)
                    emotions = torch.where(emotions == 2, torch.tensor(1), emotions)    
                emotions = emotions.to(device)
                    
                optimizer.zero_grad()
                    
                with torch.set_grad_enabled(phase == 'train'):
                    # culc forward and loss
                    emo_net_outputs, _ = emo_net(feats_list)
                    
                    loss = criterion(emo_net_outputs, emotions)
                        
                    _, preds = torch.max(emo_net_outputs, 1)
                        
                    # culc backward when training
                    if phase == 'train':
                        loss.backward()    
                        optimizer.step()
                            
                    # culc itaration loss and corrects
                    epoch_loss += loss.item() * _feats.size(0)
                    epoch_corrects += torch.sum(preds == emotions.data)
                        
                    # release memory
                    torch.cuda.empty_cache()
                
            # culc epoch loss and acc     
            epoch_loss = epoch_loss / len(dataloaders[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders[phase].dataset)
            epoch_acc = epoch_acc.item()
            
            # save log with tensorboard
            if phase == "train":
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['test_loss'].append(epoch_loss)
                history['test_acc'].append(epoch_acc)
               
            # display epoch loss and acc    
            if epoch == 0 or epoch % config.display == 0:
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                
            # save parameters
            torch.save(emo_net.state_dict(), save_each_path_dir + '/' + config.target_emo + '_' + f'emo_net.pth')
    
    print()
    print(f"----- Training of {config.target_emo} estimator finished! (fold{config.fold}) -----")
    
    history = pd.DataFrame(history)
    save_history_dir = config.write_res_prefix + config.run_name + f'/history/fold{config.fold}'
    if not os.path.exists(save_history_dir):
        os.makedirs(save_history_dir)
    history.to_csv(save_history_dir + '/' + config.target_emo + '_' + 'history.csv')
    
    
    return save_res_dir
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model configuration
    parser.add_argument('--run_name', type=str, default='default_run', help='run name')
    parser.add_argument('--window_size', type=int, default=30, help='window size')
    parser.add_argument('--use_feat_list', nargs='*', type=int, required=True, help='select feature. 0:AU, 1:Gaze, 2:HP')
    parser.add_argument('--emo_num', type=int, default=2, help='number of emotion')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads in multiheadattention')
    parser.add_argument('--d_hid', type=int, default=2048, help='dimension of transformer feedforward network')
    parser.add_argument('--num_layers', type=int, default=6, help='number of transformer encoder layers')
    parser.add_argument('--dropout', type=float, default=None, help='dropout rate')
    parser.add_argument('--wo_pe', type=str2bool, default=False, help='without positional encoding')
    parser.add_argument('--pool_type', type=str, default='mean', choices=['mean', 'max', 'cls', 'mean+max', 'att'], help='pooling type')
    parser.add_argument('--target_emo', type=str, default='comfort', choices=['comfort', 'discomfort'], help='target emotion')
    
    # training configuration
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay for SGD optimizer')
    parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--display', type=int, default=1, help='iteration gaps for displaying')
    parser.add_argument('--val_phase', type=str2bool, default=False, help='phase')
    
    # path configuration
    parser.add_argument('--write_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/Emotion_Estimator-snapshots/PIMD_A/', help='write path prefix')
    parser.add_argument('--write_res_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_labels_wsize30-ssize15.csv', help='labels directory')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_video_name_list_wsize30-ssize15.csv', help='video name list directory')
    parser.add_argument('--au_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/JAANet_feature.pkl', help='au feats directory')
    parser.add_argument('--gaze_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_feature.pkl', help='gaze feats directory')
    parser.add_argument('--hp_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_feature.pkl', help='hp feats directory')

    config = parser.parse_args()
    
    save_res_dir = main(config)
    config_dict = vars(config)
    with open(save_res_dir + '/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)