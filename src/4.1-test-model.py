"""
* This script is used to test the model that was trained in '4.1-train-model.py'.    
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from utils.util import torch_fix_seed, get_video_name_list, str2bool, standardize_feature, convert_label_to_binary
from dataset import ConcatDataset, SeqFeatList
from networks import EmotionEstimator

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

def main(config):
    # fix seed
    torch_fix_seed()
    
    # define device
    device = torch.device("cuda:" + config.gpu_id if torch.cuda.is_available() else "cpu")
    
    # define use_feat_list
    use_feat_list = {'AU':config.use_feat_list[0], 'Gaze':config.use_feat_list[1], 'HP':config.use_feat_list[2]}
    
    # define dataset and dataloader
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
    
    test_dataset = ConcatDataset(datasets_temp)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    # define pretrained model
    trained_path_dir = config.load_path_prefix + config.run_name + f'/epoch{config.target_epoch}' + '/fold' + str(config.fold)
    
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
                window_size=config.window_size,
            )
            
        if config.freeze_stream == True:
            au_stream_path = config.load_path_prefix + config.au_run_name + f'/epoch{config.au_epoch}/fold{config.fold}/' + config.target_emo + '_' + 'emo_net.pth'
        else:
            au_stream_path = trained_path_dir + "/" + config.target_emo + "_" + 'au_stream.pth'
        au_stream.load_state_dict(torch.load(au_stream_path))
        for param in au_stream.parameters():
            param.requires_grad = False
        au_stream.to(device)
        au_stream.eval()
    
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
            gaze_stream_path = config.load_path_prefix + config.gaze_run_name + f'/epoch{config.gaze_epoch}/fold{config.fold}/' + config.target_emo + '_' + 'emo_net.pth'
        else:
            gaze_stream_path = trained_path_dir + "/" + config.target_emo + "_" + 'gaze_stream.pth'
        gaze_stream.load_state_dict(torch.load(gaze_stream_path))
        for param in gaze_stream.parameters():
            param.requires_grad = False
        gaze_stream.to(device)
        gaze_stream.eval()
            
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
            hp_stream_path = config.load_path_prefix + config.hp_run_name + f'/epoch{config.hp_epoch}/fold{config.fold}/' + config.target_emo + '_' + 'emo_net.pth'
        else:
            hp_stream_path = trained_path_dir + "/" + config.target_emo + "_" + 'hp_stream.pth'
        hp_stream.load_state_dict(torch.load(hp_stream_path))
        for param in hp_stream.parameters():
            param.requires_grad = False
        hp_stream.to(device)
        hp_stream.eval()
    
    #* Mixture
    if config.summation == True:
        integrated_input_dim = config.integrate_dim
    else:
        integrated_input_dim = config.integrate_dim * sum(use_feat_list.values())
    
    if config.integrate_point == 'mid':
        emo_net = EmotionEstimator.MLPClassifier(
            input_dim=integrated_input_dim,
            hidden_dims=config.integrated_hidden_dims,
            num_classes=config.emo_num,
            dropout=config.dropout,
            activation=config.activation,
            summation=config.summation,
            ew_product=config.ew_product,
            arith_mean=config.arith_mean
        )
        emo_net.load_state_dict(torch.load(trained_path_dir + "/" + config.target_emo + "_" + 'emo_net.pth'))
        emo_net.to(device)
        emo_net.eval()
    
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
        stream_mixer.load_state_dict(torch.load(trained_path_dir + "/" + config.target_emo + "_" + 'stream_mixer.pth'))
        stream_mixer.to(device)
        stream_mixer.eval()
        
    print()
    print(f"----- Start Test... (fold{config.fold}-epoch{config.target_epoch}) -----")
    
    emo_temp_list = []
    emo_gt_list = []
    emo_pred_list= []
    emo_posterior_list = []
    img_path_list = []
    if config.is_stream_mixer == True:
        attention_weights_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            
            # get batch data
            count = 0
            feats_list = []
            logits_list = []
            if use_feat_list['AU'] == 1:
                au_feats, img_paths, emotions = batch[count]
                
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
                    gaze_feats, img_paths, emotions = batch[count]
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
                    hp_feats, img_paths, emotions = batch[count]
                else:
                    hp_feats, _, _ = batch[count]
                        
                hp_feats = hp_feats.to(device)           
 
                hp_feats = hp_feats.transpose(1, 2) 
                
                hp_stream_logits, hp_stream_outputs = hp_stream([hp_feats])
                
                feats_list.append(hp_stream_outputs)
                logits_list.append(hp_stream_logits)
                count += 1
            
            emo_temp_list += emotions.tolist()
            img_path_list += img_paths
            emotions = convert_label_to_binary(emotions, config.target_emo)    
            emotions = emotions.to(device)
            
            # forward
            if config.is_standardization == True:
                    feats_list = standardize_feature(feats_list)
            
            if config.is_stream_mixer == True:
                if config.stream_mixer_input == 'mid':
                    attention_weights = stream_mixer(feats_list)
                elif config.stream_mixer_input == 'logits':
                    attention_weights = stream_mixer(logits_list)
                attention_weights_list += attention_weights.detach().cpu().numpy().tolist()
                
                if config.integrate_point == 'mid':
                    for i in range(len(feats_list)):
                        feats_list[i] = feats_list[i] * attention_weights[:, i].unsqueeze(-1)
                elif config.integrate_point == 'logits':
                    for i in range(len(logits_list)):
                        logits_list[i] = logits_list[i] * attention_weights[:, i].unsqueeze(-1)
            
            if config.integrate_point == 'mid':
                emo_net_outputs, _ = emo_net(feats_list)
            elif config.integrate_point == 'logits':
                emo_net_outputs = torch.sum(torch.stack(logits_list), dim=0)
            
            emo_net_outputs = torch.softmax(emo_net_outputs, dim=1)
            _, emo_pred = torch.max(emo_net_outputs, dim=1)
                
            # save outputs
            if device == 'cpu':
                emo_pred_list += emo_pred.detach().numpy().tolist()
                emo_gt_list += emotions.detach().numpy().tolist()
                emo_posterior_list += emo_net_outputs[:,1].detach().numpy().tolist()
            else:  
                emo_pred_list += emo_pred.detach().cpu().numpy().tolist()
                emo_gt_list += emotions.detach().cpu().numpy().tolist()
                emo_posterior_list += emo_net_outputs[:,1].detach().cpu().numpy().tolist()
        
            # release memory
            torch.cuda.empty_cache()

    print()
    print(f"----- Finish Test... (fold{config.fold}-epoch{config.target_epoch}) -----")
    
    # calc Evaluation metrics
    save_res_dir = None
    if config.other_run_name is not None:
        save_res_dir = config.write_res_prefix + config.other_run_name + f'/epoch{config.target_epoch}' + '/fold' + str(config.fold)
    else:
        save_res_dir = config.write_res_prefix + config.run_name + f'/epoch{config.target_epoch}' + '/fold' + str(config.fold)
    
    os.makedirs(save_res_dir, exist_ok=True)
              
    precision = precision_score(emo_gt_list, emo_pred_list)
    print("precision:{}".format(precision))
    recall = recall_score(emo_gt_list, emo_pred_list)
    print("recall:{}".format(recall))
    f1 = f1_score(emo_gt_list, emo_pred_list)
    print("f1:{}".format(f1))
    accuracy = accuracy_score(emo_gt_list, emo_pred_list)
    print("accuracy:{}".format(accuracy))
    fpr, tpr, thresholds = roc_curve(emo_gt_list, emo_posterior_list)
    roc_auc = roc_auc_score(emo_gt_list, emo_posterior_list)
    print("roc_auc:{}".format(roc_auc))
    pre, rec, _ = precision_recall_curve(emo_gt_list, emo_posterior_list)
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
        sns.heatmap(emo_cm_df, annot=True, cmap='Reds', fmt='g', annot_kws={"size": 10}, vmin=0, vmax=len(emo_gt_list))
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
        
        if config.is_stream_mixer == True:
            # save attention weights
            weight_list = []
            for i in range(len(attention_weights_list)):
                weight_list.append([img_path_list[i]] + attention_weights_list[i])
            weight_df = pd.DataFrame(weight_list, columns=["img_path"] + [f"stream{i}" for i in range(len(attention_weights_list[0]))])
            weight_df.to_csv(save_res_dir + "/" + f"{config.target_emo}_attention_weights.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model configration
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
    parser.add_argument('--is_binary', type=str2bool, default=False, help='use binary or not')
    
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
    
    # test configration
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--target_epoch', type=int, default=5, help='target epoch')
    parser.add_argument('--save_res', type=str2bool, default=True)
    parser.add_argument('--other_run_name', type=str, default=None, help='other run name')
    
    # path configration
    parser.add_argument('--load_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/Emotion_Estimator-snapshots/PIMD_A/', help='load path prefix')
    parser.add_argument('--write_res_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_labels_wsize30-ssize15.csv', help='labels directory')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_video_name_list_wsize30-ssize15.csv', help='video name list directory')
    parser.add_argument('--au_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/JAANet_feature.pkl', help='au feats directory')
    parser.add_argument('--gaze_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_pitchyaw_logits.pkl', help='gaze feats directory')
    parser.add_argument('--hp_feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_logits.pkl', help='hp feats directory')

    config = parser.parse_args()
    
    main(config)
    