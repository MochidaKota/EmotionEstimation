"""
* This script is used to train the m2m model
* many to many mapping between the input and output.
"""

import argparse
import os
import json
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from utils.util import torch_fix_seed, get_video_name_list, str2bool, convert_label_to_binary
from dataset import SeqFeatList2
from networks import EmotionEstimator

def main(config):
    # fix seed
    torch_fix_seed()
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # dataset and dataloader
    traindataset = SeqFeatList2(
        labels_path=config.labels_path,
        video_name_list=get_video_name_list(config.video_name_list_path, config.fold, 'train'),
        feats_path=config.feats_path,
        window_size=config.window_size
    )
    
    testdataset = SeqFeatList2(
        labels_path=config.labels_path,
        video_name_list=get_video_name_list(config.video_name_list_path, config.fold, 'test'),
        feats_path=config.feats_path,
        window_size=config.window_size
    )
    
    trainloader = DataLoader(
        traindataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    testloader = DataLoader(
        testdataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    # model
    input_dim = pd.read_pickle(config.feats_path).shape[1] - 1
    
    if config.arch == 'lstm':
        model = EmotionEstimator.LSTMClassifier(
            num_classes=config.num_classes,
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional
        ).to(device)
    
    elif config.arch == 'transformer':
        model = EmotionEstimator.TransformerEncoderEstimator(
            num_classes=config.num_classes,
            d_model=input_dim,
            num_heads=config.num_heads,
            d_hid=config.hidden_dim,
            num_layers=config.num_layers,
            max_seq_len=config.window_size,
            dropout=config.dropout
        ).to(device)
        
    else:
        raise ValueError('Invalid model architecture')
        
    # optimizer
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        
    # criterion
    criterion = nn.BCEWithLogitsLoss()
    
    print()
    print(f"----- Start training... (fold{config.fold}) -----")
    print()
    
    history = {'epoch':[], 'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[]}
    
    for epoch in range(config.n_epochs):
        history['epoch'].append(epoch+1)
        
        # directory to save model
        model_path_dir = config.model_path_prefix + config.run_name + f'/epoch{epoch+1}' + f'/fold{config.fold}'
        os.makedirs(model_path_dir, exist_ok=True)
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
                dataloader = trainloader
            else:
                model.eval()
                dataloader = testloader
                
            running_loss = 0.0
            running_corrects = 0
            start = time.time()
            
            for i, batch in enumerate(dataloader):
                feats, _, emos = batch
                
                feats = feats.to(device)
                
                emos = convert_label_to_binary(emos, config.target_emo)
                emos = emos.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(feats)
                    
                    outputs = outputs.view(-1, config.window_size)
                    
                    loss = criterion(outputs, emos)
                    
                    preds = torch.round(torch.sigmoid(outputs))
                        
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                    running_loss += loss.item() * emos.size(0)
                    running_corrects += torch.sum(preds == emos).item()
                    
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / (len(dataloader.dataset) * config.window_size)
            
            # store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['test_loss'].append(epoch_loss)
                history['test_acc'].append(epoch_acc)
                
            print(f'[{phase}] Epoch {epoch+1}/{config.n_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {time.time()-start:.2f}sec')
            
            # save model
            if phase == 'train':
                torch.save(model.state_dict(), model_path_dir + '/' + config.target_emo + '_' + f'emo_net.pth')
                
    print()
    print(f"----- Finish training... (fold{config.fold}) -----")
    print()
    
    # save history
    history_df = pd.DataFrame(history)
    history_path_dir = config.res_path_prefix + config.run_name + f'/history/fold{config.fold}'
    os.makedirs(history_path_dir, exist_ok=True)
    history_df.to_csv(history_path_dir + '/' + config.target_emo + '_' + 'history.csv')
    
    # save config
    res_path_rootdir = config.res_path_prefix + config.run_name
    config_dict = vars(config)
    with open(res_path_rootdir + '/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    # save torchinfo summary
    with open(res_path_rootdir + '/modelsummary.txt', 'w') as f:
        f.write(repr(summary(model, input_size=(config.batch_size, config.window_size, input_dim), verbose=0)))
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_name', type=str, default='default_run', help='run name')
    parser.add_argument('--attribute', type=str, default='emotion', choices=['emotion', 'AU_sign', 'Gaze_sign', 'HP_sign'], help='attribute')
    parser.add_argument('--target_emo', type=str, default='comfort', choices=['comfort', 'discomfort'], help='target emotion')
    parser.add_argument('--window_size', type=int, default=30, help='window size')
    parser.add_argument('--arch', type=str, default='lstm', choices=['lstm', 'transformer'], help='model architecture')
    parser.add_argument('--num_classes', type=int, default=1, help='number of emotion')
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads for transformer')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    parser.add_argument('--bidirectional', type=str2bool, default=False, help='bidirectional for LSTM')
    
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'], help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    
    parser.add_argument('--model_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/Emotion_Estimator-snapshots/PIMD_A/', help='write path prefix')
    parser.add_argument('--res_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_labels_wsize30-ssize15.csv', help='labels directory')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_video_name_list_wsize30-ssize15.csv', help='video name list directory')
    parser.add_argument('--feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/blendshapes.pkl', help='feats directory')
    
    config = parser.parse_args()
    
    main(config)