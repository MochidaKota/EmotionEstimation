'''
* this script is used to train autoencoder
'''

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.util import torch_fix_seed, str2bool, get_video_name_list
from networks import AutoEncoder
from dataset import FeatList, ConcatDataset

import pandas as pd
import json

def main(config):
    # fix random seed
    torch_fix_seed()
    
    # define device
    device = torch.device('cuda:{}'.format(config.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    # define dataset and dataloader
    train_dataset = FeatList(
        labels_path=config.labels_path,
        video_name_list=get_video_name_list(config.video_name_list_path, config.fold, 'train'),
        feats_path=config.feats_path
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
    )
    
    if config.val_phase:
        test_dataset = FeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, 'test'),
            feats_path=config.feats_path
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )
    
    # define model
    input_dim = pd.read_pickle(config.feats_path).shape[1] - 1
    
    model = AutoEncoder.FeatureAutoEncoder(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        dropout=config.dropout
    ).to(device)
    
    # define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        
    # define scheduler
    if config.is_scheduler == True:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # define loss function
    criterion = nn.MSELoss()
    
    print('Start training...')
    
    if config.val_phase:
        history = {'epoch':[], 'train_loss':[], 'test_loss':[]}
        phases = ['train', 'test']
    else:    
        history = {'epoch':[], 'train_loss':[]}
        phases = ['train']
    
    for epoch in range(config.num_epochs):
        
        history['epoch'].append(epoch+1)
        
        # define directory to save model and results
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
            
        # training and validation phase
        train_loss = 0
        test_loss = 0
        
        for phase in phases:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = test_loader
                
            running_loss = 0.0
            
            for i, batch in enumerate(dataloader):
                
                inputs, _, _ = batch
                inputs = inputs.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    
                    if phase == 'train':
                        loss.backward()   
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                
            if config.is_scheduler == True and phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloader.dataset)
            if phase == 'train':
                train_loss = epoch_loss
                history['train_loss'].append(epoch_loss)
            elif phase == 'test':
                test_loss = epoch_loss
                history['test_loss'].append(epoch_loss)
        
            # save model
            torch.save(model.state_dict(), save_each_path_dir + '/autoencoder.pth')
        
        print(f'EPOCH: {epoch+1}/{config.num_epochs}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}')
                
    print('Finished training.')
    
    # save history
    history = pd.DataFrame(history)
    save_history_dir = config.write_res_prefix + config.run_name + f'/history/fold{config.fold}'
    if not os.path.exists(save_history_dir):
        os.makedirs(save_history_dir)
    history.to_csv(save_history_dir + '/history.csv')
    
    # save config
    save_root_res_dir = config.write_res_prefix + config.run_name
    config_dict = vars(config)
    with open(save_root_res_dir + '/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model config
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--hidden_dim', type=int, default=512, help='dimension of hidden layer')
    parser.add_argument('--output_dim', type=int, default=64, help='dimension of output layer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    
    # training config
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--is_scheduler', type=str2bool, default=False, help='use scheduler or not')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--val_phase', type=str2bool, default=True, help='validation phase or not')

    # path config
    parser.add_argument('--write_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/Emotion_Estimator-snapshots/PIMD_A/', help='write path prefix')
    parser.add_argument('--write_res_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/label_video18-25_AU_onlypositive.csv', help='path to labels.csv')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/video_name_list_video18-25_AU_onlypositive.csv', help='path to video_name_list.csv')
    parser.add_argument('--feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/JAANet_feature.pkl', help='path to features')
    
    config = parser.parse_args()
    
    main(config)