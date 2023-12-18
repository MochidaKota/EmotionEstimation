"""
* This script is used to train the AU, Gaze, and HP stream.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.util import torch_fix_seed, get_video_name_list, str2bool, convert_label_to_binary
from dataset import ConcatDataset, SeqFeatList
from networks import EmotionEstimator

import json
import pandas as pd
import time

def main(config):
    # fix seed
    torch_fix_seed()
    
    # define device
    device = torch.device("cuda:" + config.gpu_id if torch.cuda.is_available() else "cpu")
    
    # define use_feat_list
    use_feat_list = {'AU':config.use_feat_list[0], 'Gaze':config.use_feat_list[1], 'HP':config.use_feat_list[2]}
    
    # define dataset and dataloader
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
    print('----- datasets loaded -----')
    
    # define networks
    training_module = {}
    
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
    
    classifier = config.classifier
    
    if classifier == 'MLP':
        emo_net = EmotionEstimator.MLPClassifier(
            num_classes=config.emo_num,
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            batchnorm=config.batchnorm
        )
    elif classifier == '1DCNN':       
        emo_net = EmotionEstimator.OneDCNNClassifier(
            num_classes=config.emo_num,
            in_channel=input_dim,
            hid_channels=config.hid_channels,
            batchnorm=config.batchnorm,
            maxpool=config.maxpool
        )
    elif classifier == 'AUStream':
        emo_net = EmotionEstimator.AUStream2(
            window_size=config.window_size,
            global_pooling=config.austream_pool_type
        ) 
        
    training_module['emo_net'] = emo_net
    
    #* Attentive Pooling Layer
    if config.pool_type == 'att':
        attentive_pooling = EmotionEstimator.AttentivePooling(
            input_dim=input_dim,
            hidden_dim=input_dim // 3
        )
        training_module['attentive_pooling'] = attentive_pooling
        
    for module in training_module.values():
        module.to(device)
    
    print()
    print('----- networks loaded -----')
    
    # define optimizer
    optimizer = None
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(
            params=[{'params': module.parameters()} for module in training_module.values()],
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            params=[{'params': module.parameters()} for module in training_module.values()],
            lr=config.lr
        )
    
    # define criterion
    criterion = nn.CrossEntropyLoss()
    
    print()
    print(f"----- Start training... (fold{config.fold}) -----")
    print()
    
    if config.val_phase == True:
        phases = ['train', 'val']
        history = {'epoch':[], 'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[]}
    else:
        phases = ['train']
        history = {'epoch':[], 'train_loss':[], 'train_acc':[]}   
        
    for epoch in range(config.n_epochs):
        
        history['epoch'].append(epoch+1) 
        
        # define directory to save model and results
        save_path_dir = config.write_path_prefix + config.run_name + f'/epoch{epoch+1}'
        os.makedirs(save_path_dir, exist_ok=True)
        save_each_path_dir = save_path_dir + '/fold' + str(config.fold)
        os.makedirs(save_each_path_dir, exist_ok=True)
            
        save_res_dir = config.write_res_prefix + config.run_name + f'/epoch{epoch+1}'
        os.makedirs(save_res_dir, exist_ok=True)
        save_each_res_dir = save_res_dir + '/fold' + str(config.fold)
        os.makedirs(save_each_res_dir, exist_ok=True)
         
        # training and validation phase
        for phase in phases:
            if phase == 'train':
                for module in training_module.values():
                    module.train()
            else:
                for module in training_module.values():
                    module.eval()
                
            running_loss = 0.0
            running_corrects = 0
            start = time.time()
                
            for i, batch in enumerate(dataloaders[phase]):
                # get batch data
                feats_list = []
                count = 0
                
                if use_feat_list['AU'] == 1:
                    au_feats, _, emotions = batch[count]
                    
                    au_feats = au_feats.to(device)
                    
                    if config.classifier == '1DCNN':
                        au_feats = au_feats.transpose(1, 2)
                        
                    elif config.classifier == 'MLP':
                        if config.pool_type == 'mean':
                            au_feats = torch.mean(au_feats, dim=1)
                        elif config.pool_type == 'max':
                            au_feats = torch.max(au_feats, dim=1)[0]
                        elif config.pool_type == 'att':
                            au_feats = training_module['attentive_pooling'](au_feats)
                        
                    feats_list.append(au_feats)
                    count += 1
                    
                if use_feat_list['Gaze'] == 1:
                    if count == 0:
                        gaze_feats, _, emotions = batch[count]
                    else:
                        gaze_feats, _, _ = batch[count]
                        
                    gaze_feats = gaze_feats.to(device)
                    
                    if classifier == '1DCNN':
                        gaze_feats = gaze_feats.transpose(1, 2)
                            
                    feats_list.append(gaze_feats)
                    count += 1
                    
                if use_feat_list['HP'] == 1:
                    if count == 0:
                        hp_feats, _, emotions = batch[count]
                    else:
                        hp_feats, _, _ = batch[count]
                        
                    hp_feats = hp_feats.to(device)
                        
                    if classifier == '1DCNN':    
                        hp_feats = hp_feats.transpose(1, 2) 
                           
                    feats_list.append(hp_feats)
                    count += 1
                
                emotions = convert_label_to_binary(emotions, config.target_emo)    
                emotions = emotions.to(device)
                    
                optimizer.zero_grad()
                    
                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    if config.classifier == 'AUStream':
                        if len(feats_list) == 1:
                            feat = feats_list[0]
                        elif len(feats_list) > 1:
                            feat = torch.cat(feats_list, dim=1)
                        
                        emo_net_outputs, _ = training_module['emo_net'](feat)
                    else:
                        emo_net_outputs, _ = training_module['emo_net'](feats_list)
                        
                    loss = criterion(emo_net_outputs, emotions)
                        
                    _, preds = torch.max(emo_net_outputs, 1)
                        
                    # backward
                    if phase == 'train':
                        loss.backward()    
                        optimizer.step()
                            
                    # sum iteration loss and acc
                    running_loss += loss.item() * emotions.size(0)
                    running_corrects += torch.sum(preds == emotions.data).double().item()
                        
                    # release memory
                    torch.cuda.empty_cache()
                
            # calc epoch loss and acc     
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            
            # store history
            if phase == "train":
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['test_loss'].append(epoch_loss)
                history['test_acc'].append(epoch_acc)
               
            # display epoch loss and acc   
            print(f"epoch:{epoch+1}/{config.n_epochs} phase:{phase} loss:{epoch_loss:.4f} acc:{epoch_acc:.4f} time:{time.time()-start:.1f}[sec]")
                
            # save parameters
            torch.save(training_module['emo_net'].state_dict(), save_each_path_dir + '/' + config.target_emo + '_' + f'emo_net.pth')
            
            if config.pool_type == 'att':
                torch.save(training_module['attentive_pooling'].state_dict(), save_each_path_dir + '/' + config.target_emo + '_' + f'attentive_pooling.pth')
    
    print()
    print(f"----- Finish training... (fold{config.fold}) -----")
    
    # save history
    history = pd.DataFrame(history)
    save_history_dir = config.write_res_prefix + config.run_name + f'/history/fold{config.fold}'
    if not os.path.exists(save_history_dir):
        os.makedirs(save_history_dir)
    history.to_csv(save_history_dir + '/' + config.target_emo + '_' + 'history.csv')
    
    # save config
    save_root_res_dir = config.write_res_prefix + config.run_name
    config_dict = vars(config)
    with open(save_root_res_dir + '/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model configuration
    parser.add_argument('--run_name', type=str, default='default_run', help='run name')
    parser.add_argument('--window_size', type=int, default=30, help='window size')
    parser.add_argument('--use_feat_list', nargs='*', type=int, required=True, help='select feature. 0:AU, 1:Gaze, 2:HP')
    parser.add_argument('--emo_num', type=int, default=2, help='number of emotion')
    parser.add_argument('--target_emo', type=str, default='comfort', choices=['comfort', 'discomfort'], help='target emotion')
    parser.add_argument('--classifier', type=str, default='1DCNN', choices=['1DCNN', 'MLP', 'AUStream'], help='classifier')
    parser.add_argument('--batchnorm', type=str2bool, default=True, help='use batchnorm or not')
    parser.add_argument('--maxpool', type=str2bool, default=False, help='use maxpool or not')
    parser.add_argument('--hid_channels', nargs='*', type=int, help='hidden channels for 1DCNN')
    parser.add_argument('--hidden_dims', nargs='*', type=int, help='hidden dim for MLP')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout for MLP')
    parser.add_argument('--pool_type', type=str, default='mean', choices=['mean', 'max', 'att'], help='pooling type')
    parser.add_argument('--austream_pool_type', type=str, default='ave', choices=['ave', 'max', 'att'], help='pooling type for AUStream')
    
    # training configuration
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'], help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay for optimizer')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--val_phase', type=str2bool, default=True, help='phase')
    
    # path configuration
    parser.add_argument('--write_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/Emotion_Estimator-snapshots/PIMD_A/', help='write path prefix')
    parser.add_argument('--write_res_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_labels_wsize30-ssize15.csv', help='labels directory')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_video_name_list_wsize30-ssize15.csv', help='video name list directory')
    parser.add_argument('--au_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/JAANet_feature.pkl', help='au feats directory')
    parser.add_argument('--gaze_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_pitchyaw_logits.pkl', help='gaze feats directory')
    parser.add_argument('--hp_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_logits.pkl', help='hp feats directory')

    config = parser.parse_args()
    
    main(config)