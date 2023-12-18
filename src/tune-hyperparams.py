import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.util import torch_fix_seed, get_video_name_list, str2bool
from dataset import ConcatDataset, AUFeatList, GazeFeatList, HPSeqFeatList
from networks import EmotionEstimator

import json
import pandas as pd
import optuna

def get_optimizer(trial, model):
    optimizer_names = ['Adam', 'SGD']
    optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
    # weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-2, log=True)
    weight_decay = trial.suggest_categorical('weight_decay', [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    
    if optimizer_name == optimizer_names[0]:
        # adam_lr = trial.suggest_float('adam_lr', 1e-5, 1e-1, log=True)
        adam_lr = trial.suggest_categorical('adam_lr', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    elif optimizer_name == optimizer_names[1]:
        # sgd_lr = trial.suggest_float('sgd_lr', 1e-5, 1e-1, log=True)
        sgd_lr = trial.suggest_categorical('sgd_lr', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        sgd_momentum = trial.suggest_float('sgd_momentum', 0.0, 1.0, step=0.1)
        optimizer = optim.SGD(model.parameters(), lr=sgd_lr, momentum=sgd_momentum, weight_decay=weight_decay)
        
    return optimizer

def get_data_loaders(trial, config, fold):
    use_feat_list = {'AU':config.use_feat_list[0], 'Gaze':config.use_feat_list[1], 'HP':config.use_feat_list[2]}
    
    datasets = {}
    dataloaders = {}
    
    datasets_temp = []
    if use_feat_list['AU'] == 1:
        au_dataset = AUFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, fold, phase='train'),
            au_feats_path=config.au_feats_path
        )
        datasets_temp.append(au_dataset)
        
    if use_feat_list['Gaze'] == 1:
        gaze_dataset = GazeFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, fold, phase='train'),
            gaze_feats_path=config.gaze_feats_path
        )
        datasets_temp.append(gaze_dataset)
        
    if use_feat_list['HP'] == 1:
        hp_dataset = HPSeqFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, fold, phase='train'),
            hp_feats_path=config.hp_feats_path,
            seq_len=config.hp_seq_len,
            seq_type=config.hp_seq_type
        )
        datasets_temp.append(hp_dataset)
    
    datasets['train'] = ConcatDataset(datasets_temp)
    
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_size=trial.suggest_categorical('batch_size', [8, 16, 32]),
        shuffle=True,
        num_workers=2,
        drop_last=trial.suggest_categorical('drop_last', [True, False])
    )
    
    datasets_temp = []
    if use_feat_list['AU'] == 1:
        au_dataset = AUFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, fold, phase='test'),
            au_feats_path=config.au_feats_path
        )
        datasets_temp.append(au_dataset)
        
    if use_feat_list['Gaze'] == 1:
        gaze_dataset = GazeFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, fold, phase='test'),
            gaze_feats_path=config.gaze_feats_path
        )
        datasets_temp.append(gaze_dataset)
        
    if use_feat_list['HP'] == 1:
        hp_dataset = HPSeqFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, fold, phase='test'),
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
    
    return dataloaders
    
    

def main(config):     
    # setting device
    device = torch.device("cuda:" + config.gpu_id if torch.cuda.is_available() else "cpu")
    
    # setting use_feat_list
    use_feat_list = {'AU':config.use_feat_list[0], 'Gaze':config.use_feat_list[1], 'HP':config.use_feat_list[2]}
    
    def objective(trial):
        # fixing seed
        torch_fix_seed()
        
        fold_num = {'comfort':7, 'discomfort':5}
        
        mean_error_rate = 0
        
        for fold in range(fold_num[config.target_emo]):
        
            dataloaders = get_data_loaders(trial, config, fold+1)
            
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
            
            num_layers = trial.suggest_categorical('num_layers', [0, 1, 2, 3, 4, 5])
            hidden_dims = []
            for i in range(num_layers):
                dim = trial.suggest_categorical('hidden_dims_' + str(i), [2048, 1024, 512, 256, 128, 64])
                hidden_dims.append(dim)
            dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5, step=0.1)
            batchnorm = trial.suggest_categorical('batch_norm', [True, False])
                
            if sum(use_feat_list.values()) == 1:
                emo_net = EmotionEstimator.EmoNet_1feature(
                    input_dim=input_dim,
                    output_dim=config.emo_num,
                    num_layers=num_layers,
                    hidden_dims=hidden_dims,
                    dropout=dropout_rate,
                    batchnorm=batchnorm
                )      
            elif sum(use_feat_list.values()) == 2:
                emo_net = EmotionEstimator.EmoNet_2feature(
                    input_dims=input_dim_list,
                    output_dim=config.emo_num,
                    num_layers=num_layers,
                    hidden_dims=hidden_dims,
                    each_feat_dim=config.each_feat_dim,
                    dropout=dropout_rate,
                    batchnorm=batchnorm,
                    weighted=config.weighted,
                    same_dim=config.same_dim,
                    summation=config.summation
                )  
            elif sum(use_feat_list.values()) == 3:
                emo_net = EmotionEstimator.EmoNet_3feature(
                    input_dims=input_dim_list,
                    output_dim=config.emo_num,
                    num_layers=num_layers,
                    hidden_dims=hidden_dims,
                    each_feat_dim=config.each_feat_dim,
                    dropout=dropout_rate,
                    batchnorm=batchnorm,
                    weighted=config.weighted,
                    same_dim=config.same_dim,
                    summation=config.summation
                )
                
            emo_net.to(device)
            
            # setting optimizer
            optimizer = get_optimizer(trial, emo_net)
            
            # setting loss function
            criterion = nn.CrossEntropyLoss()
            
            error_rate = 0
            
            for epoch in range(trial.suggest_categorical('epoch', [5, 10, 20])):
                for phase in ['train', 'val']:
                    if phase == 'train':
                        emo_net.train()
                    else:
                        emo_net.eval()
                        
                    epoch_corrects = 0 # sum of corrects in epoch
                        
                    for i, batch in enumerate(dataloaders[phase]):
                            
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
                                
                            # culc backward when training
                            if phase == 'train':
                                loss.backward()    
                                optimizer.step()
                                    
                            # culc itaration loss and corrects
                            if phase == 'val':
                                _, preds = torch.max(emo_net_outputs, 1)
                                epoch_corrects += torch.sum(preds == emotions.data)
                                epoch_acc = epoch_corrects.double() / len(dataloaders[phase].dataset)
                                epoch_acc = epoch_acc.item()
                                error_rate = 1.0 - epoch_acc
                                
                            # release memory
                            torch.cuda.empty_cache()
                            
            mean_error_rate += error_rate
        
        mean_error_rate = mean_error_rate / fold_num[config.target_emo]
        
        return mean_error_rate
    
    return objective
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model configuration.
    parser.add_argument('--run_name', type=str, default='default_run', help='run name')
    parser.add_argument('--hp_seq_type', type=str, default='pool', choices=['pool', 'diff'], help='type of HPSeqFeatList')
    parser.add_argument('--hp_seq_len', type=int, default=30, help='sequence length of HPSeqFeatList')
    parser.add_argument('--use_feat_list', nargs='*', type=int, required=True, help='select feature. 0:AU, 1:Gaze, 2:HP')
    parser.add_argument('--emo_num', type=int, default=2, help='number of emotion')
    parser.add_argument('--each_feat_dim', type=int, default=512, help='size of each_feat_dim')
    parser.add_argument('--summation', type=str2bool, default=False, help='use summation of each feature')
    parser.add_argument('--weighted', type=str2bool, default=False, help='use weight of each feature')
    parser.add_argument('--same_dim', type=str2bool, default=False, help='use same dimension of each feature')
    parser.add_argument('--target_emo', type=str, default='comfort', choices=['comfort', 'discomfort'], help='target emotion')
    
    
    # Training configuration.
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--n_trials', type=int, default=100, help='number of trials')
    
    # path configuration
    parser.add_argument('--write_res_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_fbf_labels.csv', help='labels directory')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_fbf_video_name_list.csv', help='video name list directory')
    parser.add_argument('--au_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/JAANet_feature.pkl', help='au feats directory')
    parser.add_argument('--gaze_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_feature.pkl', help='gaze feats directory')
    parser.add_argument('--hp_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_feature.pkl', help='hp feats directory')
    
    config = parser.parse_args()
    
    save_res_dir = config.write_res_prefix + config.run_name
    if not os.path.exists(save_res_dir):
        os.makedirs(save_res_dir)
    
    study = optuna.create_study()
    study.optimize(main(config), n_trials=config.n_trials)
    
    # save study.best_params in save_each_res_dir
    with open(save_res_dir + '/best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    # save study.best_value in save_each_res_dir
    with open(save_res_dir + '/best_value.txt', 'w') as f:
        f.write(str(study.best_value))
        
    # save study.trials in save_each_res_dir
    df = study.trials_dataframe()
    df.to_csv(save_res_dir + '/trials.csv')
