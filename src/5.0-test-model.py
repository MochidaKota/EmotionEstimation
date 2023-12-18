import argparse
import os

import torch
from torch.utils.data import DataLoader

from utils.util import torch_fix_seed, str2bool, get_video_name_list
from networks import AutoEncoder
from dataset import FeatList, ConcatDataset

import pandas as pd

def main(config):
    # fix random seed
    torch_fix_seed()
    
    # define device
    device = torch.device('cuda:{}'.format(config.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    # define dataset and dataloader
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
    
    # load model
    trained_path_dir = config.load_path_prefix + config.run_name + f'/epoch{config.target_epoch}' + '/fold' + str(config.fold)
    model.load_state_dict(torch.load(trained_path_dir + '/autoencoder.pth'))
    model.eval()
    
    print('Start testing...')
    
    mse_list = []
    emotion_list = []
    img_path_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs, img_paths, emotions = batch
            
            inputs = inputs.to(device)
            img_path_list += img_paths
            emotion_list += emotions.tolist()
            
            outputs = model(inputs)
            
            inputs = inputs.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            
            mse = ((inputs - outputs) ** 2).mean(axis=None)
            mse_list.append(mse)
    
    print('Finish testing.')
    
    # write results
    if config.other_run_name is not None:
        save_res_dir = config.write_res_prefix + config.other_run_name + f'/epoch{config.target_epoch}' + '/fold' + str(config.fold)
    else:
        save_res_dir = config.write_res_prefix + config.run_name + f'/epoch{config.target_epoch}' + '/fold' + str(config.fold)
    if os.path.exists(save_res_dir) == False:
        os.mkdir(save_res_dir)
    
    if config.save_res:
        res_df = pd.DataFrame({'img_path':img_path_list, 'emotion':emotion_list, 'mse':mse_list})
        res_df.to_csv(save_res_dir + '/pred.csv', index=False)        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model config
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--hidden_dim', type=int, default=512, help='dimension of hidden layer')
    parser.add_argument('--output_dim', type=int, default=64, help='dimension of output layer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    
    # test config
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--target_epoch', type=int, default=0, help='target epoch')
    parser.add_argument('--save_res', type=str2bool, default=True, help='save results or not')
    parser.add_argument('--other_run_name', type=str, default=None, help='run name of other model')

    # path config
    parser.add_argument('--load_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/Emotion_Estimator-snapshots/PIMD_A/', help='write path prefix')
    parser.add_argument('--write_res_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/label_video18-25.csv', help='path to labels.csv')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/video_name_list_video18-25.csv', help='path to video_name_list.csv')
    parser.add_argument('--feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/JAANet_feature.pkl', help='path to features')
    
    config = parser.parse_args()
    
    main(config)