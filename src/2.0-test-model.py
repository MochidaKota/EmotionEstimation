"""
* This script is used to test the model that was trained in '1.2-train-model.py'.    
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.util import torch_fix_seed, get_video_name_list, str2bool
from dataset import ConcatDataset, AUFeatList, GazeFeatList, HPSeqFeatList
from networks import EmotionEstimator

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

def main(config):
    # fixing seed
    torch_fix_seed()
    
    # setting device
    device = torch.device("cuda:" + config.gpu_id if torch.cuda.is_available() else "cpu")
    
    # setting use_feat_list
    use_feat_list = {'AU':config.use_feat_list[0], 'Gaze':config.use_feat_list[1], 'HP':config.use_feat_list[2]}
    
    # setting dataset and dataloader
    dataset_temp = []
    if use_feat_list['AU'] == 1:
        au_dataset = AUFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
            au_feats_path=config.au_feats_path
        )
        dataset_temp.append(au_dataset)
    
    if use_feat_list['Gaze'] == 1:
        gaze_dataset = GazeFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
            gaze_feats_path=config.gaze_feats_path
        )
        dataset_temp.append(gaze_dataset)
    
    if use_feat_list['HP'] == 1:
        hp_dataset = HPSeqFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
            hp_feats_path=config.hp_feats_path,
            seq_len=config.hp_seq_len,
            seq_type=config.hp_seq_type
        )
        dataset_temp.append(hp_dataset)
    
    test_dataset = ConcatDataset(dataset_temp)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    # setting pretrained model
    trained_path_dir = config.load_path_prefix + config.run_name + '/fold' + str(config.fold)
    
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
        
    emo_net.load_state_dict(torch.load(trained_path_dir + "/" + config.target_emo + "_" + 'emo_net.pth', map_location=device))
    
    emo_net.to(device)
    emo_net.eval()
        
    print()
    print(f"----- Test of {config.target_emo} estimator start! (fold{config.fold}) -----")
    emo_temp_list = []
    emo_gt_list = []
    emo_pred_list= []
    emo_posterior_list = []
    img_path_list = []
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader)):
            
            if use_feat_list == {'AU':1, 'Gaze':0, 'HP':0}:
                au_feats, img_paths, emotions, _ = batch[0]
                au_feats = au_feats.to(device)
            elif use_feat_list == {'AU':0, 'Gaze':1, 'HP':0}:
                gaze_feats, img_paths, emotions = batch[0]
                gaze_feats = gaze_feats.to(device)
            elif use_feat_list == {'AU':0, 'Gaze':0, 'HP':1}:
                hp_seq_feats, img_paths, emotions = batch[0]
                hp_seq_feats = hp_seq_feats.to(device)
            elif use_feat_list == {'AU':1, 'Gaze':1, 'HP':0}:
                au_feats, img_paths, emotions, _ = batch[0]
                gaze_feats, _, _ = batch[1]
                au_feats = au_feats.to(device)
                gaze_feats = gaze_feats.to(device)
            elif use_feat_list == {'AU':1, 'Gaze':0, 'HP':1}:
                au_feats, img_paths, emotions, _ = batch[0]
                hp_seq_feats, _, _ = batch[1]
                au_feats = au_feats.to(device)
                hp_seq_feats = hp_seq_feats.to(device)
            elif use_feat_list == {'AU':0, 'Gaze':1, 'HP':1}:
                gaze_feats, img_paths, emotions = batch[0]
                hp_seq_feats, _, _ = batch[1]
                gaze_feats = gaze_feats.to(device)
                hp_seq_feats = hp_seq_feats.to(device)
            elif use_feat_list == {'AU':1, 'Gaze':1, 'HP':1}:
                au_feats, img_paths, emotions, _ = batch[0]
                gaze_feats, _, _ = batch[1]
                hp_seq_feats, _, _ = batch[2]
                au_feats = au_feats.to(device)
                gaze_feats = gaze_feats.to(device)
                hp_seq_feats = hp_seq_feats.to(device)
                
            emo_temp_list += emotions.tolist()
            img_path_list += img_paths
            if config.target_emo == 'comfort':
                emotions = torch.where(emotions == 2, torch.tensor(0), emotions)
            elif config.target_emo == 'discomfort':
                emotions = torch.where(emotions == 1, torch.tensor(0), emotions)
                emotions = torch.where(emotions == 2, torch.tensor(1), emotions) 
            emotions = emotions.to(device)
            
            if use_feat_list['HP'] == 1:
                hp_feats = None
                if config.hp_seq_type == 'pool':
                    hp_feats_mean = torch.mean(hp_seq_feats, dim=1)
                    hp_feats_max = torch.max(hp_seq_feats, dim=1)[0]
                    hp_feats = torch.cat((hp_feats_mean, hp_feats_max), dim=1)
                elif config.hp_seq_type == 'diff':
                    hp_feats = hp_seq_feats[:, -1, :] - hp_seq_feats[:, 0, :] 
            
            # culc forward
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
            
            emo_net_outputs = torch.softmax(emo_net_outputs, dim=1)
            _, emo_pred = torch.max(emo_net_outputs, dim=1)
                
            # save outputs
            emo_pred_list += emo_pred.detach().cpu().numpy().tolist()
            emo_gt_list += emotions.detach().cpu().numpy().tolist()
            emo_posterior_list += emo_net_outputs[:,1].detach().cpu().numpy().tolist()
        
            # release memory
            torch.cuda.empty_cache()

    print()
    print(f"----- Test of {config.target_emo} estimator finish! (fold{config.fold}) -----" )
    
    # culc and save Evaluation metrics
    save_res_dir = config.write_res_prefix + config.run_name + '/fold' + str(config.fold)
    if config.add_res_dir is not None:
        save_res_dir += '/' + config.add_res_dir
        if os.path.exists(save_res_dir) == False:
            os.mkdir(save_res_dir)
              
    precision = precision_score(emo_gt_list, emo_pred_list)
    print("precision:{}".format(precision))
    recall = recall_score(emo_gt_list, emo_pred_list)
    print("recall:{}".format(recall))
    f1 = f1_score(emo_gt_list, emo_pred_list)
    print("f1:{}".format(f1))
    accuracy = accuracy_score(emo_gt_list, emo_pred_list)
    print("accuracy:{}".format(accuracy))
    fpr, tpr, thresholds = roc_curve(emo_gt_list, emo_pred_list)
    roc_auc = roc_auc_score(emo_gt_list, emo_pred_list)
    print("roc_auc:{}".format(roc_auc))
    pre, rec, _ = precision_recall_curve(emo_gt_list, emo_pred_list)
    pr_auc = auc(rec, pre)
    print("pr_auc:{}".format(pr_auc))
    
    emo_clf_report_df = pd.DataFrame([[precision, recall, f1, accuracy, roc_auc, pr_auc]], columns=["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]) 
    
    
    if config.save_res == True:
        # save each metrics
        emo_clf_report_df.to_csv(save_res_dir + "/" + f"{config.target_emo}_report.csv", index=False)
        
        # save confusion matrix
        emo_cm = confusion_matrix(emo_gt_list, emo_pred_list)
        if config.target_emo == 'comfort':
            label_list = ['not comfort', 'comfort']
        elif config.target_emo == 'discomfort':
            label_list = ['not discomfort', 'discomfort']
        emo_cm_df = pd.DataFrame(emo_cm, index=label_list, columns=label_list)
        sns.heatmap(emo_cm_df, annot=True, cmap='Reds', fmt='g', annot_kws={"size": 10})
        plt.xlabel('Pred')
        plt.ylabel('GT')
        plt.title(f'{config.target_emo} Confusion Matrix in Fold{config.fold}')
        plt.tight_layout()
        plt.savefig(save_res_dir + "/" + f"{config.target_emo}_emo_cm.png")
        plt.close()
        
        # save ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%roc_auc, marker='o')
        plt.plot(np.linspace(1, 0, len(fpr)), np.linspace(1, 0, len(fpr)), label='Random ROC curve (area = %.2f)'%0.5, linestyle = '--', color = 'gray')
        plt.legend()
        plt.title(f'{config.target_emo} ROC curve in Fold{config.fold}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_res_dir + "/" + f"{config.target_emo}_roc_curve.png")
        plt.close()
        
        # save PR curve
        plt.plot(rec, pre, label='PR curve (area = %.2f)'%pr_auc, marker='o')
        plt.legend()
        plt.title(f'{config.target_emo} PR curve in Fold{config.fold}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_res_dir + "/" + f"{config.target_emo}_pr_curve.png")
        plt.close()
        
        # save outputs of model
        pred_list = []
        for i in range(len(emo_pred_list)):
            pred_list.append([emo_temp_list[i]] + [emo_pred_list[i]] + [emo_posterior_list[i]] + [img_path_list[i]])
        pred_df = pd.DataFrame(pred_list, columns=["emo_gt","emo_pred", "emo_pos", "img_path"])
        pred_df.to_csv(save_res_dir + "/" + f"{config.target_emo}_pred.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model configration
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
    
    # test configration
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--save_res', type=str2bool, default=True)
    parser.add_argument('--add_res_dir', type=str, default=None, help='add result directory name')
    
    # path configration
    parser.add_argument('--load_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/Emotion_Estimator-snapshots/PIMD_A/', help='load path prefix')
    parser.add_argument('--write_res_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort2_2.csv', help='labels directory')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort2_2-video_name_list.csv', help='video name list directory')
    parser.add_argument('--au_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/JAANet_feature.pkl', help='au feats directory')
    parser.add_argument('--gaze_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_feature.pkl', help='gaze feats directory')
    parser.add_argument('--hp_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_feature.pkl', help='hp feats directory')
    parser.add_argument('--feats_path_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/path.csv', help='feat path list directory')

    config = parser.parse_args()
    
    main(config)
    