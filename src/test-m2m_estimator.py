import argparse
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import itertools

import torch
from torch.utils.data import DataLoader

from utils.util import torch_fix_seed, get_video_name_list, str2bool, convert_label_to_binary
from dataset import SeqFeatList2
from networks import EmotionEstimator

def main(config):
    # fix seed
    torch_fix_seed()
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # dataset and dataloader
    testdataset = SeqFeatList2(
        labels_path=config.labels_path,
        video_name_list=get_video_name_list(config.video_name_list_path, config.fold, 'test'),
        feats_path=config.feats_path,
        window_size=config.window_size
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
        )
        
    elif config.arch == 'transformer':
        model = EmotionEstimator.TransformerEncoderEstimator(
            num_classes=config.num_classes,
            d_model=input_dim,
            num_heads=config.num_heads,
            d_hid=config.hidden_dim,
            num_layers=config.num_layers,
            max_seq_len=config.window_size,
            dropout=config.dropout
        )
        
    else:
        raise ValueError('Invalid model architecture')
    
    model_path_dir = config.model_path_prefix + config.run_name + f'/epoch{config.target_epoch}' + f'/fold{config.fold}'
    model.load_state_dict(torch.load(model_path_dir + "/" + config.target_emo + "_" + 'emo_net.pth'))
    model.to(device)
    model.eval()
    
    print()
    print(f"----- Start Test...(fold{config.fold}-epoch{config.target_epoch}) -----")
    print()
    
    emo_temp_list = []
    emo_gt_list = []
    emo_pred_list = []
    emo_posterior_list = []
    img_path_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            feats, img_paths, emos = batch
            
            feats = feats.to(device)
            
            emo_temp_list += emos.tolist()
            img_path_list += img_paths
            emos = convert_label_to_binary(emos, config.target_emo)
            emos = emos.to(device)
            
            outputs = model(feats)
            outputs = outputs.view(-1, config.window_size)
            posteriors = torch.sigmoid(outputs)
            preds = torch.round(posteriors)
            
            emo_gt_list += emos.detach().cpu().numpy().tolist()
            emo_pred_list += preds.detach().cpu().numpy().tolist()
            emo_posterior_list += posteriors.detach().cpu().numpy().tolist()
            
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
    
    flt_gt_list = list(itertools.chain(*emo_gt_list))
    flt_pred_list = list(itertools.chain(*emo_pred_list))
    flt_posterior_list = list(itertools.chain(*emo_posterior_list))
    
    precision = precision_score(flt_gt_list, flt_pred_list, zero_division=0)
    print("precision:{}".format(precision))
    recall = recall_score(flt_gt_list, flt_pred_list, zero_division=0)
    print("recall:{}".format(recall))
    f1 = f1_score(flt_gt_list, flt_pred_list, zero_division=0)
    print("f1:{}".format(f1))
    accuracy = accuracy_score(flt_gt_list, flt_pred_list)
    print("accuracy:{}".format(accuracy))
    fpr, tpr, _ = roc_curve(flt_gt_list, flt_posterior_list)
    roc_auc = auc(fpr, tpr)
    print("roc_auc:{}".format(roc_auc))
    pre, rec, _ = precision_recall_curve(flt_gt_list, flt_posterior_list)
    pr_auc = auc(rec, pre)
    print("pr_auc:{}".format(pr_auc))
    
    # save each metrics
    emo_clf_report_df = pd.DataFrame([[precision, recall, f1, accuracy, roc_auc, pr_auc]], columns=["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"])
    emo_clf_report_df.to_csv(res_path_dir + "/" + f"{config.target_emo}_report.csv", index=False)
    
    # save confusion matrix
    emo_cm = confusion_matrix(flt_gt_list, flt_pred_list)
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
    plt.savefig(res_path_dir + "/" + f"{config.target_emo}_emo_cm.png")
    plt.close()
    
    # save outputs
    pred_list = []
    for i in range(len(emo_pred_list)):
        pred_list.append([emo_temp_list[i]] + [emo_pred_list[i]] + [emo_posterior_list[i]] + [img_path_list[i]])
    pred_df = pd.DataFrame(pred_list, columns=["emo_gt","emo_pred", "emo_pos", "img_path"])
    pred_df.to_csv(res_path_dir + "/" + f"{config.target_emo}_pred.csv", index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_name', type=str, default='default_run', help='run name')
    parser.add_argument('--attribute', type=str, default='emotion', choices=['emotion', 'AU_sign', 'Gaze_sign', 'HP_sign'], help='attribute')
    parser.add_argument('--target_emo', type=str, default='comfort', choices=['comfort', 'discomfort'], help='target emotion')
    parser.add_argument('--window_size', type=int, default=30, help='window size')
    parser.add_argument('--arch', type=str, default='lstm', choices=['lstm', 'transformer'], help='model architecture')
    parser.add_argument('--num_classes', type=int, default=1, help='number of emotion')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    parser.add_argument('--bidirectional', type=str2bool, default=False, help='bidirectional')
    
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--target_epoch', type=int, default=5, help='target epoch')
    parser.add_argument('--other_run_name', type=str, default=None, help='run name of other test')
    
    parser.add_argument('--model_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/Emotion_Estimator-snapshots/PIMD_A/', help='write path prefix')
    parser.add_argument('--res_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_labels_wsize30-ssize15.csv', help='labels directory')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_video_name_list_wsize30-ssize15.csv', help='video name list directory')
    parser.add_argument('--feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/blendshapes.pkl', help='feats directory')
    
    config = parser.parse_args()
    
    main(config)