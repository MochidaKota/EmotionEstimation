"""
* This script is used to test the model that was trained in '4.0-train-model.py'.    
"""

import argparse
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils.util import torch_fix_seed, get_video_name_list, str2bool, convert_label_to_binary
from dataset import SeqFeatList
from networks import EmotionEstimator

def main(config):
    # fix seed
    torch_fix_seed()
    
    # define device
    device = torch.device("cuda:" + config.gpu_id if torch.cuda.is_available() else "cpu")
    
    # define use_feat_list
    use_feat_list = {'AU':config.use_feat_list[0], 'Gaze':config.use_feat_list[1], 'HP':config.use_feat_list[2]}
    
    # define dataset and dataloader
    if use_feat_list['AU'] == 1:
        feats_path = config.au_feats_path
    if use_feat_list['Gaze'] == 1:
        feats_path = config.gaze_feats_path    
    if use_feat_list['HP'] == 1:
        feats_path = config.hp_feats_path
        
    test_dataset = SeqFeatList(
            labels_path=config.labels_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
            feats_path=feats_path,
            window_size=config.window_size
        )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    # load pretrained model
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
    
    emo_net = EmotionEstimator.MLPClassifier(
        num_classes=config.emo_num,
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        batchnorm=config.batchnorm
    )
    
    model_path_dir = config.model_path_prefix + config.run_name + f'/epoch{config.target_epoch}' + f'/fold{config.fold}'   
    emo_net.load_state_dict(torch.load(model_path_dir + "/" + config.target_emo + "_" + 'emo_net.pth'))
    emo_net.to(device)
    emo_net.eval()
    
    #* Attentive Pooling Layer
    if config.pool_type == 'att':
        attentive_pooling = EmotionEstimator.AttentivePooling(
            input_dim=input_dim,
            pool_type=config.att_pool_type
        )
        
        attentive_pooling.load_state_dict(torch.load(model_path_dir + "/" + config.target_emo + "_" + 'attentive_pooling.pth'))
        attentive_pooling.to(device)
        attentive_pooling.eval()
        
    print()
    print(f"----- Start Test...(fold{config.fold}-epoch{config.target_epoch}) -----" )
    print()
    
    emo_temp_list = []
    emo_gt_list = []
    emo_pred_list= []
    emo_posterior_list = []
    img_path_list = []
    mid_feat_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            
            # get batch data
            feats, img_paths, emotions = batch
                
            feats = feats.to(device)    
            if config.pool_type == 'avg':
                feats = torch.mean(feats, dim=1)
            elif config.pool_type == 'max':
                feats = torch.max(feats, dim=1)[0]
            elif config.pool_type == 'att':
                feats, _ = attentive_pooling(feats)
                
            emo_temp_list += emotions.tolist()
            img_path_list += img_paths
            emotions = convert_label_to_binary(emotions, config.target_emo) 
            emotions = emotions.to(device)
            
            # forward
            emo_net_outputs, mid_feat = emo_net(feats)
            
            emo_net_outputs = torch.softmax(emo_net_outputs, dim=1)
            
            _, emo_pred = torch.max(emo_net_outputs, dim=1)
                
            # save outputs
            if device == 'cpu':
                emo_pred_list += emo_pred.detach().numpy().tolist()
                emo_gt_list += emotions.detach().numpy().tolist()
                emo_posterior_list += emo_net_outputs[:,1].detach().numpy().tolist()
                mid_feat_list += mid_feat.detach().numpy().tolist()
            else:  
                emo_pred_list += emo_pred.detach().cpu().numpy().tolist()
                emo_gt_list += emotions.detach().cpu().numpy().tolist()
                emo_posterior_list += emo_net_outputs[:,1].detach().cpu().numpy().tolist()
                mid_feat_list += mid_feat.detach().cpu().numpy().tolist()
        
            # release memory
            del feats, emotions, emo_net_outputs, mid_feat
            torch.cuda.empty_cache()

    print()
    print(f"----- Finish Test...(fold{config.fold}-epoch{config.target_epoch}) -----")
    print()
    
    # calc Evaluation metrics
    res_path_dir = None
    if config.other_run_name is not None:
        res_path_dir = config.res_path_prefix + config.other_run_name + f'/epoch{config.target_epoch}' + f'/fold{config.fold}'
    else:
        res_path_dir = config.res_path_prefix + config.run_name + f'/epoch{config.target_epoch}' + f'/fold{config.fold}'
    
    os.makedirs(res_path_dir, exist_ok=True)
              
    precision = precision_score(emo_gt_list, emo_pred_list)
    print("precision:{}".format(precision))
    recall = recall_score(emo_gt_list, emo_pred_list)
    print("recall:{}".format(recall))
    f1 = f1_score(emo_gt_list, emo_pred_list)
    print("f1:{}".format(f1))
    accuracy = accuracy_score(emo_gt_list, emo_pred_list)
    print("accuracy:{}".format(accuracy))
    fpr, tpr, _ = roc_curve(emo_gt_list, emo_posterior_list)
    roc_auc = auc(fpr, tpr)
    print("roc_auc:{}".format(roc_auc))
    pre, rec, _ = precision_recall_curve(emo_gt_list, emo_posterior_list)
    pr_auc = auc(rec, pre)
    print("pr_auc:{}".format(pr_auc))
     
    
    if config.save_res == True:
        # save each metrics
        emo_clf_report_df = pd.DataFrame([[precision, recall, f1, accuracy, roc_auc, pr_auc]], columns=["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"])
        emo_clf_report_df.to_csv(res_path_dir + "/" + f"{config.target_emo}_report.csv", index=False)
        
        # save confusion matrix
        emo_cm = confusion_matrix(emo_gt_list, emo_pred_list)
        if config.target_emo == 'comfort':
            label_list = ['not comfort', 'comfort']
        elif config.target_emo == 'discomfort':
            label_list = ['not discomfort', 'discomfort']
        emo_cm_df = pd.DataFrame(emo_cm, index=label_list, columns=label_list)
        sns.heatmap(emo_cm_df, annot=True, cmap='Reds', fmt='g', annot_kws={"size": 10}, vmin=0, vmax=len(emo_gt_list))
        plt.xlabel('Pred')
        plt.ylabel('GT')
        plt.title(f'{config.target_emo} Confusion Matrix in Fold{config.fold}')
        plt.tight_layout()
        plt.savefig(res_path_dir + "/" + f"{config.target_emo}_emo_cm.png")
        plt.close()
        
        # save outputs
        pred_list = []
        for i in range(len(emo_pred_list)):
            pred_list.append([emo_temp_list[i]] + [emo_pred_list[i]] + [emo_posterior_list[i]] + [img_path_list[i]])
        pred_df = pd.DataFrame(pred_list, columns=["emo_gt","emo_pred", "emo_pos", "img_path"])
        pred_df.to_csv(res_path_dir + "/" + f"{config.target_emo}_pred.csv", index=False)
        
    if config.save_feat == True:
        feat_list = []
        for i in range(len(mid_feat_list)):
            feat_list.append([img_path_list[i]] + [emo_temp_list[i]] + [emo_pred_list[i]] + mid_feat_list[i])
        feat_df = pd.DataFrame(feat_list, columns=["img_path", "emo_gt", "emo_pred"] + [f"feat{i}" for i in range(len(mid_feat_list[0]))])
        feat_df.to_csv(res_path_dir + "/" + f"{config.target_emo}_feat.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model configration
    parser.add_argument('--run_name', type=str, default='default_run', help='run name')
    parser.add_argument('--window_size', type=int, default=30, help='window size')
    parser.add_argument('--use_feat_list', nargs='*', type=int, required=True, help='select feature. 0:AU, 1:Gaze, 2:HP')
    parser.add_argument('--emo_num', type=int, default=2, help='number of emotion')
    parser.add_argument('--target_emo', type=str, default='comfort', choices=['comfort', 'discomfort'], help='target emotion')
    parser.add_argument('--batchnorm', type=str2bool, default=True, help='use batchnorm or not')
    parser.add_argument('--hidden_dims', nargs='*', type=int, help='hidden dim for MLP')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout for MLP')
    parser.add_argument('--pool_type', type=str, default='avg', choices=['avg', 'max', 'att'], help='pooling type')
    parser.add_argument('--att_pool_type', type=str, default='woLi', choices=['base', 'woLi', 'Li', 'MLP'], help='attention pooling type')
    
    # test configration
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--target_epoch', type=int, default=5, help='target epoch')
    parser.add_argument('--save_res', type=str2bool, default=True)
    parser.add_argument('--save_feat', type=str2bool, default=False)
    parser.add_argument('--other_run_name', type=str, default=None, help='run name of other test')
    
    # path configration
    parser.add_argument('--model_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/Emotion_Estimator-snapshots/PIMD_A/', help='load path prefix')
    parser.add_argument('--res_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_labels_wsize30-ssize15.csv', help='labels directory')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_video_name_list_wsize30-ssize15.csv', help='video name list directory')
    parser.add_argument('--au_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/JAANet_feature.pkl', help='au feats directory')
    parser.add_argument('--gaze_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_pitchyaw_logits.pkl', help='gaze feats directory')
    parser.add_argument('--hp_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_logits.pkl', help='hp feats directory')

    config = parser.parse_args()
    
    main(config)
    