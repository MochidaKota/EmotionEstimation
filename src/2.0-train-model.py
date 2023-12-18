"""
* This script is used to train the AU-Gaze-HP model.
* AU:feat, Gaze:feat, HP:seq_feat
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.util import torch_fix_seed, get_video_name_list, str2bool
from dataset import ConcatDataset, AUFeatList, GazeFeatList, HPSeqFeatList
from networks import EmotionEstimator

from tqdm import tqdm
import json
import pandas as pd

def main(config):
    # fixing seed
    torch_fix_seed()
    
    # setting directory
    save_path_dir = config.write_path_prefix + config.run_name
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    save_each_path_dir = save_path_dir + '/fold' + str(config.fold)
    if not os.path.exists(save_each_path_dir):
        os.makedirs(save_each_path_dir)
    
    save_res_dir = config.write_res_prefix + config.run_name
    if not os.path.exists(save_res_dir):
        os.makedirs(save_res_dir)
    save_each_res_dir = save_res_dir + '/fold' + str(config.fold)
    if not os.path.exists(save_each_res_dir):
        os.makedirs(save_each_res_dir)

    # setting device
    device = torch.device("cuda:" + config.gpu_id if torch.cuda.is_available() else "cpu")
    
    # setting use_feat_list
    use_feat_list = {'AU':config.use_feat_list[0], 'Gaze':config.use_feat_list[1], 'HP':config.use_feat_list[2]}
    
    # setting dataset and dataloader
    datasets = {}
    dataloaders = {}
    
    datasets_temp = []
    if use_feat_list['AU'] == 1:
        au_dataset = AUFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='train'),
            au_feats_path=config.au_feats_path
        )
        datasets_temp.append(au_dataset)
        
    if use_feat_list['Gaze'] == 1:
        gaze_dataset = GazeFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='train'),
            gaze_feats_path=config.gaze_feats_path
        )
        datasets_temp.append(gaze_dataset)
        
    if use_feat_list['HP'] == 1:
        hp_dataset = HPSeqFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='train'),
            hp_feats_path=config.hp_feats_path,
            seq_len=config.hp_seq_len,
            seq_type=config.hp_seq_type
        )
        datasets_temp.append(hp_dataset)
    
    datasets['train'] = ConcatDataset(datasets_temp)
    
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    if config.val_phase == True:
        datasets_temp = []
        if use_feat_list['AU'] == 1:
            au_dataset = AUFeatList(
                labels_path=config.labels_path,
                video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
                au_feats_path=config.au_feats_path
            )
            datasets_temp.append(au_dataset)
            
        if use_feat_list['Gaze'] == 1:
            gaze_dataset = GazeFeatList(
                labels_path=config.labels_path,
                video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
                gaze_feats_path=config.gaze_feats_path
            )
            datasets_temp.append(gaze_dataset)
            
        if use_feat_list['HP'] == 1:
            hp_dataset = HPSeqFeatList(
                labels_path=config.labels_path,
                video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
                hp_feats_path=config.hp_feats_path,
                seq_len=config.hp_seq_len,
                seq_type=config.hp_seq_type
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
    #* Emotion Estimator
    input_dim_list = []
    if use_feat_list['AU'] == 1:
        input_dim_list.append(12000)
    if use_feat_list['Gaze'] == 1:
        input_dim_list.append(2048)
    if use_feat_list['HP'] == 1:
        if config.hp_seq_type == 'pool':
            input_dim_list.append(4096)
        elif config.hp_seq_type == 'diff':
            input_dim_list.append(2048)
    if len(input_dim_list) == 1:
        input_dim = input_dim_list[0]
    
    if sum(use_feat_list.values()) == 1:
        emo_net = EmotionEstimator.EmoNet_1feature(
            input_dim=input_dim,
            output_dim=config.emo_num,
            num_layers=config.num_layers,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            batchnorm=config.batchnorm
        )      
    elif sum(use_feat_list.values()) == 2:
        emo_net = EmotionEstimator.EmoNet_2feature(
            input_dims=input_dim_list,
            output_dim=config.emo_num,
            num_layers=config.num_layers,
            hidden_dims=config.hidden_dims,
            each_feat_dim=config.each_feat_dim,
            dropout=config.dropout,
            batchnorm=config.batchnorm,
            weighted=config.weighted,
            same_dim=config.same_dim,
            summation=config.summation
        )  
    elif sum(use_feat_list.values()) == 3:
        emo_net = EmotionEstimator.EmoNet_3feature(
            input_dims=input_dim_list,
            output_dim=config.emo_num,
            num_layers=config.num_layers,
            hidden_dims=config.hidden_dims,
            each_feat_dim=config.each_feat_dim,
            dropout=config.dropout,
            batchnorm=config.batchnorm,
            weighted=config.weighted,
            same_dim=config.same_dim,
            summation=config.summation
        )
    
    print(emo_net)
      
    emo_net.to(device)
    
    print()
    print('----- networks loaded -----')
    
    # setting optimizer
    optimizer = None
    if config.optimizer == 'SGD':
        optimizer = optim.SGD(
            emo_net.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(
            emo_net.parameters(),
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
        # Each epoch has only training phase    
        for phase in phases:
            if phase == 'train':
                emo_net.train()
            else:
                emo_net.eval()
                
            epoch_loss = 0.0  # sum of loss in epoch
            epoch_corrects = 0 # sum of corrects in epoch
                
            for i, batch in tqdm(enumerate(dataloaders[phase])):
                # emo_label_list = {'others': 0, 'comfort': 1, 'discomfort': 2}
                    
                if use_feat_list == {'AU':1, 'Gaze':0, 'HP':0}:
                    au_feats, _, emotions, _ = batch[0]
                    au_feats = au_feats.to(device)
                elif use_feat_list == {'AU':0, 'Gaze':1, 'HP':0}:
                    gaze_feats, _, emotions = batch[0]
                    gaze_feats = gaze_feats.to(device)
                elif use_feat_list == {'AU':0, 'Gaze':0, 'HP':1}:
                    hp_seq_feats, _, emotions = batch[0]
                    hp_seq_feats = hp_seq_feats.to(device)
                elif use_feat_list == {'AU':1, 'Gaze':1, 'HP':0}:
                    au_feats, _, emotions, _ = batch[0]
                    gaze_feats, _, _ = batch[1]
                    au_feats = au_feats.to(device)
                    gaze_feats = gaze_feats.to(device)
                elif use_feat_list == {'AU':1, 'Gaze':0, 'HP':1}:
                    au_feats, _, emotions, _ = batch[0]
                    hp_seq_feats, _, _ = batch[1]
                    au_feats = au_feats.to(device)
                    hp_seq_feats = hp_seq_feats.to(device)
                elif use_feat_list == {'AU':0, 'Gaze':1, 'HP':1}:
                    gaze_feats, _, emotions = batch[0]
                    hp_seq_feats, _, _ = batch[1]
                    gaze_feats = gaze_feats.to(device)
                    hp_seq_feats = hp_seq_feats.to(device)
                elif use_feat_list == {'AU':1, 'Gaze':1, 'HP':1}:
                    au_feats, _, emotions, _ = batch[0]
                    gaze_feats, _, _ = batch[1]
                    hp_seq_feats, _, _ = batch[2]
                    au_feats = au_feats.to(device)
                    gaze_feats = gaze_feats.to(device)
                    hp_seq_feats = hp_seq_feats.to(device)
                    
                _feats = None
                if use_feat_list['AU'] == 1:
                    if _feats is None:
                        _feats = au_feats 
                if use_feat_list['Gaze'] == 1:
                    if _feats is None:
                        _feats = gaze_feats
                if use_feat_list['HP'] == 1:
                    hp_feats = None
                    if config.hp_seq_type == 'pool':
                        hp_feats_mean = torch.mean(hp_seq_feats, dim=1)
                        hp_feats_max = torch.max(hp_seq_feats, dim=1)[0]
                        hp_feats = torch.cat((hp_feats_mean, hp_feats_max), dim=1)
                    elif config.hp_seq_type == 'diff':
                        hp_feats = hp_seq_feats[:, -1, :] - hp_seq_feats[:, 0, :]  
                    if _feats is None:
                        _feats = hp_feats
                
                if config.target_emo == 'comfort':
                    emotions = torch.where(emotions == 2, torch.tensor(0), emotions)
                elif config.target_emo == 'discomfort':
                    emotions = torch.where(emotions == 1, torch.tensor(0), emotions)
                    emotions = torch.where(emotions == 2, torch.tensor(1), emotions)    
                emotions = emotions.to(device)
                    
                optimizer.zero_grad()
                    
                with torch.set_grad_enabled(phase == 'train'):
                    # culc forward and loss
                    if use_feat_list == {'AU':1, 'Gaze':0, 'HP':0}:
                        emo_net_outputs = emo_net(au_feats)
                    elif use_feat_list == {'AU':0, 'Gaze':1, 'HP':0}:
                        emo_net_outputs = emo_net(gaze_feats)
                    elif use_feat_list == {'AU':0, 'Gaze':0, 'HP':1}:
                        emo_net_outputs = emo_net(hp_feats)
                    elif use_feat_list == {'AU':1, 'Gaze':1, 'HP':0}:
                        emo_net_outputs = emo_net(au_feats, gaze_feats)
                    elif use_feat_list == {'AU':1, 'Gaze':0, 'HP':1}:
                        emo_net_outputs = emo_net(au_feats, hp_feats)
                    elif use_feat_list == {'AU':0, 'Gaze':1, 'HP':1}:
                        emo_net_outputs = emo_net(gaze_feats, hp_feats)
                    elif use_feat_list == {'AU':1, 'Gaze':1, 'HP':1}:
                        emo_net_outputs = emo_net(au_feats, gaze_feats, hp_feats)
                    
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
                
            # save parameters of last epoch model
            if epoch == config.n_epochs - 1:
                torch.save(emo_net.state_dict(), save_each_path_dir + '/' + config.target_emo + '_' + 'emo_net.pth')
    
    print()
    print(f"----- Training of {config.target_emo} estimator finished! (fold{config.fold}) -----")
    
    history = pd.DataFrame(history)
    history.to_csv(save_each_res_dir + '/' + config.target_emo + '_' + 'history.csv')
    
    return save_res_dir
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model configuration
    parser.add_argument('--run_name', type=str, default='default_run', help='run name')
    parser.add_argument('--hp_seq_type', type=str, default='pool', choices=['pool', 'diff'], help='type of HPSeqFeatList')
    parser.add_argument('--hp_seq_len', type=int, default=30, help='sequence length of HPSeqFeatList')
    parser.add_argument('--use_feat_list', nargs='*', type=int, required=True, help='select feature. 0:AU, 1:Gaze, 2:HP')
    parser.add_argument('--emo_num', type=int, default=2, help='number of emotion')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--hidden_dims', nargs='*', type=int, required=True, help='size of hidden dimensions')
    parser.add_argument('--each_feat_dim', type=int, default=512, help='size of each_feat_dim')
    parser.add_argument('--dropout', type=float, default=None, help='dropout rate')
    parser.add_argument('--batchnorm', type=str2bool, default=False, help='use batch normalization')
    parser.add_argument('--summation', type=str2bool, default=False, help='use summation of each feature')
    parser.add_argument('--weighted', type=str2bool, default=False, help='use weight of each feature')
    parser.add_argument('--same_dim', type=str2bool, default=False, help='use same dimension of each feature')
    parser.add_argument('--target_emo', type=str, default='comfort', choices=['comfort', 'discomfort'], help='target emotion')
    
    # training configuration
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
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
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort2_2.csv', help='labels directory')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort2_2-video_name_list.csv', help='video name list directory')
    parser.add_argument('--au_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/JAANet_feature.pkl', help='au feats directory')
    parser.add_argument('--gaze_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_feature.pkl', help='gaze feats directory')
    parser.add_argument('--hp_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_feature.pkl', help='hp feats directory')
    parser.add_argument('--feats_path_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/path.csv', help='feat path list directory')

    config = parser.parse_args()
    
    save_res_dir = main(config)
    config_dict = vars(config)
    with open(save_res_dir + '/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)