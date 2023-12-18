"""
* This script is used to train the AU-Gaze-HP model.
* AU:Image, Gaze:Image, HP:Image
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from utils.util import torch_fix_seed, get_video_name_list, str2bool
from dataset import ConcatDataset, AUImageList, GazeImageList, HPImageList
from preprocess import JAANet_ImageTransform, L2CSNet_ImageTransform, sixDRepNet_ImageTransform
from networks import EmotionEstimator, JAANet_networks, L2CSNet_networks, sixDRepNet_networks

from tqdm import tqdm
import pandas as pd
import json

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
    print()
    print(f"use device -> {device}")
    
    # setting use_feat_list
    use_feat_list = {'AU':config.use_feat_list[0], 'Gaze':config.use_feat_list[1], 'HP':config.use_feat_list[2]}
    
    # setting dataset and dataloader
    datasets = {}
    dataloaders = {}
    
    datasets_temp = []
    if use_feat_list['AU'] == 1:
        au_dataset = AUImageList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='train'),
            au_transform=JAANet_ImageTransform(phase='train')
        )
        datasets_temp.append(au_dataset)
        
    if use_feat_list['Gaze'] == 1:
        gaze_dataset = GazeImageList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='train'),
            gaze_transform=L2CSNet_ImageTransform(phase='train')
        )
        datasets_temp.append(gaze_dataset)
        
    if use_feat_list['HP'] == 1:
        hp_dataset = HPImageList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='train'),
            hp_transform=sixDRepNet_ImageTransform(phase='train')
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
            au_dataset = AUImageList(
                labels_path=config.labels_path,
                video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
                au_transform=JAANet_ImageTransform(phase='test')
            )
            datasets_temp.append(au_dataset)
            
        if use_feat_list['Gaze'] == 1:
            gaze_dataset = GazeImageList(
                labels_path=config.labels_path,
                video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
                gaze_transform=L2CSNet_ImageTransform(phase='test')
            )
            datasets_temp.append(gaze_dataset)
            
        if use_feat_list['HP'] == 1:
            hp_dataset = HPImageList(
                labels_path=config.labels_path,
                video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
                hp_transform=sixDRepNet_ImageTransform(phase='test')
            )
            datasets_temp.append(hp_dataset)
        
        datasets['val'] = ConcatDataset(datasets_temp)
        
        dataloaders['val'] = DataLoader(
            datasets['val'],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )
    
    print()
    print('----- Dataset and Dataloader Setting -----')
    
    # setting networks
    freeze_modules = []
    #* JAANet(AU Estimator)
    if use_feat_list['AU'] == 1:
        region_learning = JAANet_networks.network_dict['HMRegionLearning'](input_dim=3, unit_dim=8)
        region_learning.load_state_dict(torch.load('/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/JAANet-snapshots/region_learning.pth'))
        freeze_modules.append(region_learning)
        
        align_net = JAANet_networks.network_dict['AlignNet'](crop_size=176, map_size=44, au_num=12, land_num=49, input_dim=64, fill_coeff=0.56)
        align_net.load_state_dict(torch.load('/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/JAANet-snapshots/align_net.pth'))
        freeze_modules.append(align_net)
        
        local_attention_refine = JAANet_networks.network_dict['LocalAttentionRefine'](au_num=12, unit_dim=8)
        local_attention_refine.load_state_dict(torch.load('/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/JAANet-snapshots/local_attention_refine.pth'))
        freeze_modules.append(local_attention_refine)
        
        local_au_net = JAANet_networks.network_dict['LocalAUNetv1'](au_num=12, input_dim=64, unit_dim=8)
        local_au_net.load_state_dict(torch.load('/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/JAANet-snapshots/local_au_net.pth'))
        freeze_modules.append(local_au_net)
        
        global_au_feat = JAANet_networks.network_dict['HLFeatExtractor'](input_dim=64, unit_dim=8)           
        global_au_feat.load_state_dict(torch.load('/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/JAANet-snapshots/global_au_feat.pth'))
        freeze_modules.append(global_au_feat)
    
    #* L2CSNet(Gaze Estimator)
    #* its architecture is same as ResNet50
    if use_feat_list['Gaze'] == 1:
        gaze_feat_extractor = L2CSNet_networks.L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
        gaze_feat_extractor.load_state_dict(torch.load('/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/L2CSNet-snapshots/Gaze360/L2CSNet_gaze360.pkl'))
        freeze_modules.append(gaze_feat_extractor)
        
    #* 6DRepNet(Head Pose Estimator)
    if use_feat_list['HP'] == 1:
        hp_feat_extractor = sixDRepNet_networks.SixDRepNet(backbone_name='RepVGG-B1g2', backbone_file='', deploy=True, pretrained=False)
        hp_feat_extractor.load_state_dict(torch.load("/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/6DRepNet-snapshots/6DRepNet_300W_LP_AFLW2000.pth"))
        freeze_modules.append(hp_feat_extractor)
        
    #* Emotion Estimator
    # hp_dim_list = {'mean': 2048, 'max': 2048, 'mean_and_max': 4096}
    input_dim_list = []
    if use_feat_list['AU'] == 1:
        input_dim_list.append(12000)
    if use_feat_list['Gaze'] == 1:
        input_dim_list.append(2048)
    if use_feat_list['HP'] == 1:
        # input_dim_list.append(hp_dim_list[config.hp_pooling])
        input_dim_list.append(2048)
    if len(input_dim_list) == 1:
        input_dim = input_dim_list[0]
    
    if sum(use_feat_list.values()) == 1:
        emo_net = EmotionEstimator.EmoNet_1feature(
            input_dim=input_dim,
            output_dim=config.emo_num,
            num_layers=config.num_layers,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout
        )      
    elif sum(use_feat_list.values()) == 2:
        emo_net = EmotionEstimator.EmoNet_2feature(
            input_dims=input_dim_list,
            output_dim=config.emo_num,
            num_layers=config.num_layers,
            hidden_dims=config.hidden_dims,
            each_feat_dim=config.each_feat_dim,
            dropout=config.dropout,
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
            weighted=config.weighted,
            same_dim=config.same_dim,
            summation=config.summation
        )
       
    for module in freeze_modules:
        for param in module.parameters():
            param.requires_grad = False
            
        module.to(device)
        module.eval()
    
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
        history['epoch'].append(epoch + 1)
                   
        # Each epoch has only training phase    
        for phase in phases:
            if phase == 'train':
                emo_net.train()
            else:
                emo_net.eval()
              
            epoch_loss = 0.0  # sum of loss in epoch
            epoch_corrects = 0 # sum of corrects in epoch
                
            for i, batch in tqdm(enumerate(dataloaders[phase])):
                
                if use_feat_list == {'AU':1, 'Gaze':0, 'HP':0}:
                    au_imgs, _, emotions, _ = batch[0]
                    au_imgs = au_imgs.to(device)
                elif use_feat_list == {'AU':0, 'Gaze':1, 'HP':0}:
                    gaze_imgs, _, emotions = batch[0]
                    gaze_imgs = gaze_imgs.to(device)
                elif use_feat_list == {'AU':0, 'Gaze':0, 'HP':1}:
                    hp_imgs, _, emotions = batch[0]
                    hp_imgs = hp_imgs.to(device)
                elif use_feat_list == {'AU':1, 'Gaze':1, 'HP':0}:
                    au_imgs, _, emotions, _ = batch[0]
                    gaze_imgs, _, _ = batch[1]
                    au_imgs = au_imgs.to(device)
                    gaze_imgs = gaze_imgs.to(device)
                elif use_feat_list == {'AU':1, 'Gaze':0, 'HP':1}:
                    au_imgs, _, emotions, _ = batch[0]
                    hp_imgs, _, _ = batch[1]
                    au_imgs = au_imgs.to(device)
                    hp_imgs = hp_imgs.to(device)
                elif use_feat_list == {'AU':0, 'Gaze':1, 'HP':1}:
                    gaze_imgs, _, emotions = batch[0]
                    hp_imgs, _, _ = batch[1]
                    gaze_imgs = gaze_imgs.to(device)
                    hp_imgs = hp_imgs.to(device)
                elif use_feat_list == {'AU':1, 'Gaze':1, 'HP':1}:
                    au_imgs, _, emotions, _ = batch[0]
                    gaze_imgs, _, _ = batch[1]
                    hp_imgs, _, _ = batch[2]
                    au_imgs = au_imgs.to(device)
                    gaze_imgs = gaze_imgs.to(device)
                    hp_imgs = hp_imgs.to(device)
                
                if config.target_emo == 'comfort':
                    emotions = torch.where(emotions == 2, torch.tensor(0), emotions)
                elif config.target_emo == 'discomfort':
                    emotions = torch.where(emotions == 1, torch.tensor(0), emotions)
                    emotions = torch.where(emotions == 2, torch.tensor(1), emotions)    
                emotions = emotions.to(device)
                    
                optimizer.zero_grad()
                
                with torch.no_grad():
                    _feats = None
                    if use_feat_list['AU'] == 1:
                        region_feat = region_learning(au_imgs)
                        align_feat, align_output, aus_map = align_net(region_feat)
                        aus_map = aus_map.to(device)
                        output_aus_map = local_attention_refine(aus_map.detach())
                        local_au_out_feat = local_au_net(region_feat, output_aus_map)
                        global_au_out_feat = global_au_feat(region_feat)
                        au_feats = torch.cat((align_feat, global_au_out_feat, local_au_out_feat), dim=1)
                        au_feats = au_feats.view(au_feats.size(0), -1)
                        if _feats is None:
                            _feats = au_feats
                        
                    if use_feat_list['Gaze'] == 1:
                        _, _, gaze_feats = gaze_feat_extractor(gaze_imgs)
                        if _feats is None:
                            _feats = gaze_feats
                        
                    if use_feat_list['HP'] == 1:
                        _, hp_feats = hp_feat_extractor(hp_imgs)  
                        if _feats is None:
                            _feats = hp_feats
                    
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
            
            # save epoch loss and acc
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
    parser.add_argument('--hp_seq_len', type=int, default=30, help='sequence length of HPSeqFeatList')
    parser.add_argument('--hp_pooling', type=str, default='mean_and_max', choices=['mean', 'max', 'mean_and_max'], help='pooling method of HP')
    parser.add_argument('--use_feat_list', nargs='*', type=int, required=True, help='select feature. 0:AU, 1:Gaze, 2:HP')
    parser.add_argument('--emo_num', type=int, default=2, help='number of emotion')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--hidden_dims', nargs='*', type=int, required=True, help='size of hidden dimensions')
    parser.add_argument('--each_feat_dim', type=int, default=512, help='size of each_feat_dim')
    parser.add_argument('--dropout', type=float, default=None, help='dropout rate')
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
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay for optimizer')
    parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--display', type=int, default=1, help='iteration gaps for displaying')
    parser.add_argument('--val_phase', type=str2bool, default=False, help='phase')
    
    # path configuration
    parser.add_argument('--write_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/Emotion_Estimator-snapshots/PIMD_A/', help='write path prefix')
    parser.add_argument('--write_res_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort2_2.csv', help='labels directory')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort2_2-video_name_list.csv', help='video name list directory')
    parser.add_argument('--hp_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_feature.pkl', help='hp feats directory')

    config = parser.parse_args()
    
    save_res_dir = main(config)
    config_dict = vars(config)
    with open(save_res_dir + '/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)