"""
* This script is used to train integrated model.
* the method of integrating AU, Gaze, HP is concatenating or sammation each stream's middle layer output.
* AU:seq_feat, Gaze:seq_feat, HP:seq_feat
* stream baseline
*   AU -> 2-1_?_a_mean_ws30-ss5-adamw, epoch10
*   Gaze -> 2-2_?_g_womp-logits-nch2_ws30-ss5-adamw, epoch10
*   HP -> 2-2_?_h_womp-logits-nch2.2_ws30-ss5-adamw, epoch5
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.util import torch_fix_seed, get_video_name_list, str2bool, standardize_feature, convert_label_to_binary
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
    
    #* AU stream
    if use_feat_list['AU'] == 1:
        if config.au_stream == 'MLP':
            au_stream = EmotionEstimator.MLPClassifier(
                input_dim=config.au_input_dim,
                hidden_dims=[2048, 512],
                num_classes=config.emo_num,
                dropout=0.1
            )
        elif config.au_stream == '1DCNN':
            au_stream = EmotionEstimator.OneDCNNClassifier(
                in_channel=config.au_input_dim,
                hid_channels=[2048, 512],
                num_classes=config.emo_num,
                maxpool=False
            )
        elif config.au_stream == 'AUStream':
            au_stream = EmotionEstimator.AUStream2(
                window_size=config.window_size
            )
        
        if config.freeze_stream == True:
            au_stream_path = config.write_path_prefix + config.au_run_name + f'/epoch{config.au_epoch}/fold{config.fold}/' + config.target_emo + '_' + 'emo_net.pth'
            au_stream.load_state_dict(torch.load(au_stream_path))
            for param in au_stream.parameters():
                param.requires_grad = False
            au_stream.to(device)
            au_stream.eval()
        else:
           training_module['au_stream'] = au_stream 
    
    #* Gaze stream
    if use_feat_list['Gaze'] == 1:
        if config.gaze_stream == '1DCNN':
            gaze_stream = EmotionEstimator.OneDCNNClassifier(
                in_channel=config.gaze_input_dim,
                hid_channels=[256, 512],
                num_classes=config.emo_num,
                maxpool=False
            )
        
        if config.freeze_stream == True:
            gaze_stream_path = config.write_path_prefix + config.gaze_run_name + f'/epoch{config.gaze_epoch}/fold{config.fold}/' + config.target_emo + '_' + 'emo_net.pth'
            gaze_stream.load_state_dict(torch.load(gaze_stream_path))
            for param in gaze_stream.parameters():
                param.requires_grad = False
            gaze_stream.to(device)
            gaze_stream.eval()
        else:
            training_module['gaze_stream'] = gaze_stream
            
    #* HP stream
    if use_feat_list['HP'] == 1:
        if config.hp_stream == '1DCNN':
            hp_stream = EmotionEstimator.OneDCNNClassifier(
                in_channel=config.hp_input_dim,
                hid_channels=[8, 512],
                num_classes=config.emo_num,
                maxpool=False
            )
        
        if config.freeze_stream == True:
            hp_stream_path = config.write_path_prefix + config.hp_run_name + f'/epoch{config.hp_epoch}/fold{config.fold}/' + config.target_emo + '_' + 'emo_net.pth'
            hp_stream.load_state_dict(torch.load(hp_stream_path))
            for param in hp_stream.parameters():
                param.requires_grad = False
            hp_stream.to(device)
            hp_stream.eval()
        else:
            training_module['hp_stream'] = hp_stream
    
    #* Mixture
    #TODO integrate_dimの使い方を変える
    if config.summation == True:
        integrated_input_dim = config.integrate_dim
    else:
        integrated_input_dim = config.integrate_dim * sum(use_feat_list.values())
    
    if config.integrate_point == 'mid':
        emo_net = EmotionEstimator.MLPClassifier(
            input_dim=integrated_input_dim,
            hidden_dims=config.integrated_hidden_dims,
            num_classes=config.emo_num,
            activation=config.activation,
            dropout=config.dropout,
            summation=config.summation,
            ew_product=config.ew_product,
            arith_mean=config.arith_mean
        )
        training_module['emo_net'] = emo_net
    
    if config.is_stream_mixer == True:
        
        if config.stream_mixer_input == 'mid':
            stream_mixer_input_dim = config.integrate_dim
        elif config.stream_mixer_input == 'logits':
            stream_mixer_input_dim = config.emo_num
              
        stream_mixer = EmotionEstimator.StreamMixer(
            feats_dim=stream_mixer_input_dim,
            num_feats=sum(use_feat_list.values()),
            hidden_dims=config.stream_mixer_hidden_dims,
            is_binary=config.is_binary
        )
        training_module['stream_mixer'] = stream_mixer
    
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
                count = 0
                feats_list = []
                logits_list = []
                
                if use_feat_list['AU'] == 1:
                    au_feats, _, emotions = batch[count]
                    
                    au_feats = au_feats.to(device)
                    
                    if config.au_stream == 'MLP':
                        au_feats = torch.mean(au_feats, dim=1)
                    elif config.au_stream == '1DCNN':
                        au_feats = au_feats.transpose(1, 2)
                    
                    if config.au_stream == 'AUStream':
                        au_stream_logits, au_stream_outputs = au_stream(au_feats)
                    else:
                        au_stream_logits, au_stream_outputs = au_stream([au_feats])
    
                    feats_list.append(au_stream_outputs)
                    logits_list.append(au_stream_logits)
                    count += 1
                    
                if use_feat_list['Gaze'] == 1:
                    if count == 0:
                        gaze_feats, _, emotions = batch[count]
                    else:
                        gaze_feats, _, _ = batch[count]
                
                    gaze_feats = gaze_feats.to(device)                
                
                    gaze_feats = gaze_feats.transpose(1, 2)
                    
                    gaze_stream_logits, gaze_stream_outputs = gaze_stream([gaze_feats])
                    
                    feats_list.append(gaze_stream_outputs)
                    logits_list.append(gaze_stream_logits)
                    count += 1
                    
                if use_feat_list['HP'] == 1:
                    if count == 0:
                        hp_feats, _, emotions = batch[count]
                    else:
                        hp_feats, _, _ = batch[count]
                    
                    hp_feats = hp_feats.to(device)
                          
                    hp_feats = hp_feats.transpose(1, 2) 
                    
                    hp_stream_logits, hp_stream_outputs = hp_stream([hp_feats])
                    
                    feats_list.append(hp_stream_outputs)
                    logits_list.append(hp_stream_logits)
                    count += 1
                
                emotions = convert_label_to_binary(emotions, config.target_emo)   
                emotions = emotions.to(device)
                    
                optimizer.zero_grad()
                    
                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    
                    if config.is_standardization == True:
                        feats_list = standardize_feature(feats_list)
                    
                    if config.is_stream_mixer == True:
                        if config.stream_mixer_input == 'mid':
                            attention_weights = training_module['stream_mixer'](feats_list)
                        elif config.stream_mixer_input == 'logits':
                            attention_weights = training_module['stream_mixer'](logits_list)
                        
                        if config.integrate_point == 'mid':
                            for i in range(len(feats_list)):
                                feats_list[i] = feats_list[i] * attention_weights[:, i].unsqueeze(-1)
                        elif config.integrate_point == 'logits':
                            for i in range(len(logits_list)):
                                logits_list[i] = logits_list[i] * attention_weights[:, i].unsqueeze(-1)
                    
                    if config.integrate_point == 'mid':
                        emo_net_outputs, _ = training_module['emo_net'](feats_list)
                    elif config.integrate_point == 'logits':
                        emo_net_outputs = torch.sum(torch.stack(logits_list), dim=0)
                    
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
            if config.integrate_point == 'mid':
                torch.save(training_module['emo_net'].state_dict(), save_each_path_dir + '/' + config.target_emo + '_' + f'emo_net.pth')
            
            if config.freeze_stream == False:
                torch.save(training_module['au_stream'].state_dict(), save_each_path_dir + '/' + config.target_emo + '_' + f'au_stream.pth')
                torch.save(training_module['gaze_stream'].state_dict(), save_each_path_dir + '/' + config.target_emo + '_' + f'gaze_stream.pth')
                torch.save(training_module['hp_stream'].state_dict(), save_each_path_dir + '/' + config.target_emo + '_' + f'hp_stream.pth')
            
            if config.is_stream_mixer == True:
                torch.save(training_module['stream_mixer'].state_dict(), save_each_path_dir + '/' + config.target_emo + '_' + f'stream_mixer.pth')
    
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
    parser.add_argument('--integrated_hidden_dims', nargs='*', type=int, help='hidden dim for MLP')
    parser.add_argument('--stream_mixer_hidden_dims', nargs='*', type=int, help='hidden dim for StreamMixer')
    parser.add_argument('--stream_mixer_input', type=str, default='mid', choices=['mid', 'logits'], help='input for StreamMixer')
    parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'LeakyReLU'], help='activation for MLP')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout for MLP')
    parser.add_argument('--summation', type=str2bool, default=False, help='use sammation or not')
    parser.add_argument('--maxpool', type=str2bool, default=False, help='use maxpool or not')
    parser.add_argument('--is_stream_mixer', type=str2bool, default=False, help='use stream mixer or not')
    parser.add_argument('--ew_product', type=str2bool, default=False, help='use element-wise product or not')
    parser.add_argument('--arith_mean', type=str2bool, default=False, help='use arithmetic mean or not')
    parser.add_argument('--is_standardization', type=str2bool, default=False, help='use standardization or not')
    parser.add_argument('--freeze_stream', type=str2bool, default=True, help='freeze stream or not')
    parser.add_argument('--integrate_point', type=str, default='mid', choices=['mid', 'logits'], help='integrate point')
    parser.add_argument('--is_binary', type=str2bool, default=False, help='use binary attention or not')
    
    parser.add_argument('--integrate_dim', type=int, default=512, help='integrate dim')
    parser.add_argument('--au_stream', type=str, default='MLP', choices=['MLP', '1DCNN', 'AUStream'], help='stream')
    parser.add_argument('--au_run_name', type=str, default='4_c_a', help='run name')
    parser.add_argument('--au_epoch', type=int, default=10, help='epoch')
    parser.add_argument('--au_input_dim', type=int, default=12000, help='input dim')
    parser.add_argument('--gaze_stream', type=str, default='1DCNN', choices=['1DCNN'], help='stream')
    parser.add_argument('--gaze_run_name', type=str, default='4_c_g', help='run name')
    parser.add_argument('--gaze_epoch', type=int, default=10, help='epoch')
    parser.add_argument('--gaze_input_dim', type=int, default=180, help='input dim')
    parser.add_argument('--hp_stream', type=str, default='1DCNN', choices=['1DCNN'], help='stream')
    parser.add_argument('--hp_run_name', type=str, default='4_c_h', help='run name')
    parser.add_argument('--hp_epoch', type=int, default=5, help='epoch')
    parser.add_argument('--hp_input_dim', type=int, default=6, help='input dim')

    # training configuration
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'], help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay for SGD optimizer')
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