import glob
import os
import pandas as pd
import argparse
from util import str2bool
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

def main(config):
    save_res_dir = config.write_res_prefix + config.run_name + '/' + f'epoch{config.target_epoch}'
    if os.path.exists(save_res_dir) == False:
        save_res_dir = config.write_res_prefix + config.run_name
    report = pd.DataFrame()
    pred = pd.DataFrame()
    
    for i in range(len(glob.glob(save_res_dir + "/fold*"))):
        fold_dir = save_res_dir + "/fold{}".format(i + 1)
        if config.only == True:
            print(f"fold{i + 1} only_{config.target_emo}_video")
            fold_report = pd.read_csv(fold_dir + f"/only_{config.target_emo}_video/{config.target_emo}_report.csv")
            fold_pred = pd.read_csv(fold_dir + f"/only_{config.target_emo}_video/{config.target_emo}_pred.csv")
        else:
            fold_report = pd.read_csv(fold_dir + f"/{config.target_emo}_report.csv")
            fold_pred = pd.read_csv(fold_dir + f"/{config.target_emo}_pred.csv")
        report = pd.concat([report, fold_report], axis=0)
        pred = pd.concat([pred, fold_pred], axis=0)
    
    report.index = ["fold{}".format(i + 1) for i in range(report.shape[0])]
    
    report_mean = pd.DataFrame(report.mean()).T
    report_mean.index = ["mean"]
    report = pd.concat([report, report_mean], axis=0)
    report.columns = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]
    
    pred = pred.sort_values('img_path')
    pred = pred.reset_index(drop=True)
    gt_list = []
    if config.target_emo == "comfort":
        # copy pred["emo_gt"] to gt_list
        gt_list = pred["emo_gt"].copy()
        # replace 2 to 0
        gt_list = gt_list.replace(2, 0)
    elif config.target_emo == "discomfort":
        # copy pred["emo_gt"] to gt_list
        gt_list = pred["emo_gt"].copy()
        # replace 1 to 0
        gt_list = gt_list.replace(1, 0)
        # replace 2 to 1
        gt_list = gt_list.replace(2, 1)
        
    # culc metrics gt_list and pred["emo_pred"]
    # roc_auc, pr_auc is calculated by pred["emo_pos"]
    acc = accuracy_score(gt_list, pred["emo_pred"])
    precision = precision_score(gt_list, pred["emo_pred"])
    recall = recall_score(gt_list, pred["emo_pred"])
    f1 = f1_score(gt_list, pred["emo_pred"])
    roc_auc = roc_auc_score(gt_list, pred["emo_pos"])
    pre, rec, _ = precision_recall_curve(gt_list, pred["emo_pos"])
    pr_auc = auc(rec, pre)
    
    # save confusion matrix
    emo_cm = confusion_matrix(gt_list, pred["emo_pred"])
    if config.target_emo == 'comfort':
        label_list = ['not comfort', 'comfort']
    elif config.target_emo == 'discomfort':
        label_list = ['not discomfort', 'discomfort']
    emo_cm_df = pd.DataFrame(emo_cm, index=label_list, columns=label_list)
    sns.heatmap(emo_cm_df, annot=True, cmap='Reds', fmt='g', annot_kws={"size": 10}, vmin=0, vmax=len(pred))
    plt.xlabel('Pred')
    plt.ylabel('GT')
    plt.tight_layout()
    plt.savefig(save_res_dir + "/" + f"{config.target_emo}_cm.png")
    plt.close()
    
    # save pred
    if config.only == True:
        pred.to_csv(save_res_dir + f"/only_{config.target_emo}_video_pred.csv", index=False)
    else:
        pred.to_csv(save_res_dir + f"/{config.target_emo}_pred_all.csv", index=False)
        
    # save metrics as a csv file
    metrics = pd.DataFrame([precision, recall, f1, acc, roc_auc, pr_auc]).T
    metrics.columns = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]
    
    if config.only == True:
        metrics.to_csv(save_res_dir + f"/only_{config.target_emo}_video_metrics(micro_ave).csv", index=False)
    else:
        metrics.to_csv(save_res_dir + f"/{config.target_emo}_metrics(micro_ave).csv", index=False)
    
       
    # save report
    if config.only == True:
        report.to_csv(save_res_dir + f"/only_{config.target_emo}_video_report.csv")
    else:
        report.to_csv(save_res_dir + f"/{config.target_emo}_report.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_res_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--target_epoch", type=str, default="best")
    parser.add_argument("--target_emo", type=str, default="comfort")
    parser.add_argument("--only", type=str2bool, default=False)
    config = parser.parse_args()
    
    print()
    print("------calc cv result------")
    
    main(config)