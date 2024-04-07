import glob
import os
import pandas as pd
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import itertools

def main(config):
    res_path_dir = config.res_path_prefix + config.run_name + '/' + f'epoch{config.target_epoch}'
    os.makedirs(res_path_dir, exist_ok=True)
    report = pd.DataFrame()
    pred = pd.DataFrame()
    
    for i in range(len(glob.glob(res_path_dir + "/fold*"))):
        fold_dir = res_path_dir + "/fold{}".format(i + 1)
        fold_report = pd.read_csv(fold_dir + f"/{config.target_emo}_report.csv")
        fold_pred = pd.read_csv(fold_dir + f"/{config.target_emo}_pred.csv", converters={'emo_gt': eval, 'emo_pred': eval, 'emo_pos': eval})
        report = pd.concat([report, fold_report], axis=0)
        pred = pd.concat([pred, fold_pred], axis=0)
    
    pred = pred.sort_values('img_path')
    pred = pred.reset_index(drop=True)
    
    emo_gt_list = list(itertools.chain(*pred["emo_gt"].tolist()))
    emo_pred_list = list(itertools.chain(*pred["emo_pred"].tolist()))
    emo_pos_list = list(itertools.chain(*pred["emo_pos"].tolist()))
    seq_pred_list = pred["seq_pred"].tolist()
    seq_gt_list = pred["seq_gt"].tolist()
    
    if config.target_emo == "comfort":
        # replace 2 to 0
        emo_gt_list = [0.0 if i == 2.0 else i for i in emo_gt_list]
    elif config.target_emo == "discomfort":
        # replace 1 to 0
        emo_gt_list = [0.0 if i == 1.0 else i for i in emo_gt_list]
        # replace 2 to 1
        emo_gt_list = [1.0 if i == 2.0 else i for i in emo_gt_list]
        
    # culc metrics gt_list and pred["emo_pred"]
    # roc_auc, pr_auc is calculated by pred["emo_pos"]
    acc = accuracy_score(emo_gt_list, emo_pred_list)
    precision = precision_score(emo_gt_list, emo_pred_list)
    recall = recall_score(emo_gt_list, emo_pred_list)
    f1 = f1_score(emo_gt_list, emo_pred_list)
    fpr, tpr, _ = roc_curve(emo_gt_list, emo_pos_list)
    roc_auc = auc(fpr, tpr)
    pre, rec, _ = precision_recall_curve(emo_gt_list, emo_pos_list)
    pr_auc = auc(rec, pre)
    
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
    plt.tight_layout()
    plt.savefig(res_path_dir + "/" + f"confusion_matrix.png")
    plt.close()
    
    # save roc curve
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%roc_auc)
    plt.plot(np.linspace(1, 0, len(fpr)), np.linspace(1, 0, len(fpr)), label='Random ROC curve (area = %.2f)'%0.5, linestyle = '--', color = 'gray')
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.tight_layout()
    plt.savefig(res_path_dir + "/" + f"roc_curve.png")
    plt.close()
    
    # save pr curve
    plt.plot(rec, pre, label='PR curve (area = %.2f)'%pr_auc)
    plt.legend()
    plt.title('PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.tight_layout()
    plt.savefig(res_path_dir + "/" + f"pr_curve.png")
    plt.close()
    
    # save pred
    pred.to_csv(res_path_dir + f"/pred_all.csv", index=False)
        
    # save metrics
    metrics = pd.DataFrame([precision, recall, f1, acc, roc_auc, pr_auc]).T
    metrics.columns = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]
    metrics.to_csv(res_path_dir + f"/metrics.csv", index=False)
    
    # save metrics(macro_avg)
    report.index = ["fold{}".format(i + 1) for i in range(report.shape[0])]
    report_mean = pd.DataFrame(report.mean()).T
    report_mean.index = ["mean"]
    report = pd.concat([report, report_mean], axis=0)
    report.columns = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc", "seq_precision", "seq_recall", "seq_f1", "seq_accuracy"]
    report.to_csv(res_path_dir + f"/metrics(macro_avg).csv")
    
    # save metrics(sequence_avg)
    seq_precision = precision_score(seq_gt_list, seq_pred_list)
    seq_recall = recall_score(seq_gt_list, seq_pred_list)
    seq_f1 = f1_score(seq_gt_list, seq_pred_list)
    seq_acc = accuracy_score(seq_gt_list, seq_pred_list)
    
    seq_metrics = pd.DataFrame([seq_precision, seq_recall, seq_f1, seq_acc]).T
    seq_metrics.columns = ["precision", "recall", "f1", "accuracy"]
    seq_metrics.to_csv(res_path_dir + f"/metrics(sequence_avg).csv", index=False)
    
    print()
    print(f'-----calcuation result for epoch{config.target_epoch}-----')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--res_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--target_epoch", type=str, default="best")
    parser.add_argument("--target_emo", type=str, default="comfort")
    
    config = parser.parse_args()
    
    main(config)