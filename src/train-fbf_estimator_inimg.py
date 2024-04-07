"""
*This script is used to train the FBF estimator model.
* one to one mapping between the input and output.
* The main difference is that the input is a single image.
"""

import argparse
import os
import json
import pandas as pd
import time

# use GPU id 1,2,3
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
import torchvision

from utils.util import torch_fix_seed, get_video_name_list, str2bool, convert_label_to_binary
from dataset import ImageList
from preprocess import JAANet_ImageTransform

def main(config):
    # fix seed
    torch_fix_seed()
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # dataset and dataloader
    traindataset = ImageList(
        labels_path=config.labels_path,
        video_name_list=get_video_name_list(config.video_name_list_path, config.fold, 'train'),
        transform=JAANet_ImageTransform(phase='train', size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        attribute=config.attribute
    )
    
    testdataset = ImageList(
        labels_path=config.labels_path,
        video_name_list=get_video_name_list(config.video_name_list_path, config.fold, 'test'),
        transform=JAANet_ImageTransform(phase='test', size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        attribute=config.attribute
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
    
    print(get_video_name_list(config.video_name_list_path, config.fold, 'train'))
    print(f'train class count:{traindataset.get_class_sample_count()}')
    print(get_video_name_list(config.video_name_list_path, config.fold, 'test'))
    print(f'test class count:{testdataset.get_class_sample_count()}')
    
    # network
    if config.arch == 'resnet':
        emo_net = torchvision.models.resnet50(pretrained=config.pretrained)
        emo_net.fc = nn.Linear(emo_net.fc.in_features, config.num_classes)
    elif config.arch == 'densenet':
        # emo_net = torchvision.models.densenet121(pretrained=config.pretrained)
        emo_net = torchvision.models.densenet169(pretrained=config.pretrained)
        emo_net.classifier = nn.Linear(emo_net.classifier.in_features, config.num_classes)
    elif config.arch == 'vgg':
        emo_net = torchvision.models.vgg16(pretrained=config.pretrained)
        emo_net.classifier[6] = nn.Linear(emo_net.classifier[6].in_features, config.num_classes)
    elif config.arch == 'vit':
        emo_net = torchvision.models.vit_b_16(pretrained=config.pretrained)
        emo_net.heads[0] = nn.Linear(emo_net.heads[0].in_features, config.num_classes)
    
    # freeze except for the last layer
    if config.freeze:
        for param in emo_net.parameters():
            param.requires_grad = False
        if config.arch == 'resnet':
            emo_net.fc.weight.requires_grad = True
            emo_net.fc.bias.requires_grad = True
        elif config.arch == 'densenet':
            emo_net.classifier.weight.requires_grad = True
            emo_net.classifier.bias.requires_grad = True
        elif config.arch == 'vgg':
            emo_net.classifier[6].weight.requires_grad = True
            emo_net.classifier[6].bias.requires_grad = True
        elif config.arch == 'vit':
            emo_net.heads[0].weight.requires_grad = True
            emo_net.heads[0].bias.requires_grad = True
    
    # use multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        emo_net = nn.DataParallel(emo_net)
        
    emo_net.to(device)
        
    # optimizer
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(emo_net.module.parameters(), lr=config.lr)
    elif config.optimizer == 'AdamW':
        optimizer = optim.AdamW(emo_net.module.parameters(), lr=config.lr)
        
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
                emo_net.train()
                dataloader = trainloader
            else:
                emo_net.eval()
                dataloader = testloader
                
            running_loss = 0.0
            running_corrects = 0
            start_time = time.time()
            
            for i, batch in enumerate(dataloader):
                imgs, _, emotions = batch
                
                imgs = imgs.to('cuda')
                
                emotions = convert_label_to_binary(emotions, config.target_emo)
                emotions = emotions.to('cuda')
                    
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = emo_net(imgs)
                    
                    loss = criterion(outputs.view(-1), emotions.float())
                    
                    preds = torch.round(torch.sigmoid(outputs)).view(-1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    running_loss += loss.item() * imgs.size(0)
                    running_corrects += torch.sum(preds == emotions.data).item()
                    print(f'[{phase}] batch{i+1}/{len(dataloader)} Loss: {loss.item():.4f} Acc: {torch.sum(preds == emotions.data).item()/len(imgs):.4f}', end='\r')
                    
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / len(dataloader.dataset)
            
            # store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['test_loss'].append(epoch_loss)
                history['test_acc'].append(epoch_acc)
                
            print(f'[{phase}] Epoch {epoch+1}/{config.n_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {time.time()-start_time:.2f}sec')
            
            # save model
            if phase == 'train':
                torch.save(emo_net.module.state_dict(), model_path_dir + '/' + config.target_emo + '_' + f'emo_net.pth')
                
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
        f.write(repr(summary(emo_net, input_size=(config.batch_size, 3, 224, 224), verbose=0)))
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_name', type=str, default='default_run', help='run name')
    parser.add_argument('--attribute', type=str, default='emotion', choices=['emotion', 'AU_sign', 'Gaze_sign', 'HP_sign'], help='attribute')
    parser.add_argument('--target_emo', type=str, default='comfort', choices=['comfort', 'discomfort'], help='target emotion')
    parser.add_argument('--num_classes', type=int, default=1, help='number of emotion')
    parser.add_argument('--arch', type=str, default='resnet', choices=['resnet', 'densenet', 'vgg', 'vit'], help='network architecture')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='use pretrained model')
    parser.add_argument('--freeze', type=str2bool, default=False, help='freeze except for the last layer')
    
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
    
    config = parser.parse_args()
    
    main(config)