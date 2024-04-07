import argparse
import os
import json
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from utils.util import torch_fix_seed, get_video_name_list, str2bool, convert_label_to_binary
from dataset import FeatList

def main(config):
    # fix seed
    torch_fix_seed()
    
    # define dataset
    train_dataset = FeatList(
        labels_path=config.labels_path,
        video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='train'),
        feats_path=config.feats_path,
        attribute=config.attribute
    )
    train_path_list = []
    for _, path, _ in train_dataset:
        train_path_list.append(path)
    
    test_dataset = FeatList(
        labels_path=config.labels_path,
        video_name_list=get_video_name_list(config.video_name_list_path, config.fold, phase='test'),
        feats_path=config.feats_path,
        attribute=config.attribute
    )
    test_path_list = []
    for _, path, _ in test_dataset:
        test_path_list.append(path)
        
    # define feature and label
    base_df = pd.read_csv(config.labels_path)
    
    X_train = base_df[base_df['img_path'].isin(train_path_list)].iloc[:, 24:]
    X_train = X_train.reset_index(drop=True)
    
    # select features
    # blendshape_names = ['_neutral','browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight','cheekPuff','cheekSquintLeft','cheekSquintRight','eyeBlinkLeft','eyeBlinkRight',
    #                     'eyeLookDownLeft','eyeLookDownRight','eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight','eyeLookUpLeft','eyeLookUpRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft',
    #                     'eyeWideRight','jawForward','jawLeft','jawOpen','jawRight','mouthClose','mouthDimpleLeft','mouthDimpleRight','mouthFrownLeft','mouthFrownRight','mouthFunnel','mouthLeft','mouthLowerDownLeft',
    #                     'mouthLowerDownRight','mouthPressLeft','mouthPressRight','mouthPucker','mouthRight','mouthRollLower','mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthSmileLeft','mouthSmileRight',
    #                     'mouthStretchLeft','mouthStretchRight','mouthUpperUpLeft','mouthUpperUpRight','noseSneerLeft','noseSneerRight'
    #                     ]
    if config.select_features:
        bls_list = ['mouthLeft', 'mouthRight', 'jawLeft', 'jawRight']
        X_train = X_train[bls_list]
    
    y_train = base_df[base_df['img_path'].isin(train_path_list)][config.attribute]
    y_train = y_train.reset_index(drop=True)
    y_train = convert_label_to_binary(y_train, config.target_emo)
    
    # print each class number
    print("train class number")
    print(y_train.value_counts())
    print()
    
    # random under sampling
    if config.is_under_sampling:
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    
        # print each class number
        print("train class number after random under sampling")
        print(y_train.value_counts())
        print()
    elif config.is_over_sampling:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
        # print each class number
        print("train class number after smote")
        print(y_train.value_counts())
        print()
    
    X_test = base_df[base_df['img_path'].isin(test_path_list)].iloc[:, 24:]
    X_test = X_test.reset_index(drop=True)
    if config.select_features:
        X_test = X_test[bls_list]
    
    y_test = base_df[base_df['img_path'].isin(test_path_list)][config.attribute]
    y_test = y_test.reset_index(drop=True)
    y_test = convert_label_to_binary(y_test, config.target_emo)
    
    # print each class number
    print("test class number")
    print(y_test.value_counts())
    print()
    
    # define model
    # if config.is_class_weight:
    #     class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    #     class_weight_array = np.array([class_weight[0] if i == 0 else class_weight[1] for i in y_train])
    #     lgb_train = lgb.Dataset(X_train, y_train, weight=class_weight_array)
    # else:
    #     lgb_train = lgb.Dataset(X_train, y_train)
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'num_leaves': config.num_leaves,
        'max_depth': config.max_depth,
        'learning_rate': config.learning_rate,  
        'random_state': 42,
        'is_unbalance': config.is_class_weight, 
    }
    verbose_eval = 0
    
    # train model
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=config.num_boost_round
    )
    
    # define directory to save model
    model_path_dir = config.model_path_prefix + config.run_name + f'/epoch{0}' + f'/fold{config.fold}'
    os.makedirs(model_path_dir, exist_ok=True)
    
    # save model
    gbm.save_model(model_path_dir + "/" + f"{config.target_emo}_lightgbm.txt", num_iteration=gbm.best_iteration)
    
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_binary = [1 if i > 0.5 else 0 for i in y_pred]
    
    # calculate metrics and save results
    res_path_dir = None
    if config.other_run_name is not None:
        res_path_dir = config.res_path_prefix + config.other_run_name + f'/epoch{0}' + f'/fold{config.fold}'
    else:
        res_path_dir = config.res_path_prefix + config.run_name + f'/epoch{0}' + f'/fold{config.fold}'
    
    os.makedirs(res_path_dir, exist_ok=True)
    
    emo_temp_list = base_df[base_df['img_path'].isin(test_path_list)][config.attribute].tolist()
    emo_gt_list = y_test.tolist()
    emo_pred_list= y_pred_binary
    emo_posterior_list = y_pred
    img_path_list = test_path_list
    
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
    
    emo_clf_report_df = pd.DataFrame([[precision, recall, f1, accuracy, roc_auc, pr_auc]], columns=["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"])
    emo_clf_report_df.to_csv(res_path_dir + "/" + f"{config.target_emo}_report.csv", index=False)
    
    pred_list = []
    for i in range(len(emo_pred_list)):
        pred_list.append([emo_temp_list[i]] + [emo_pred_list[i]] + [emo_posterior_list[i]] + [img_path_list[i]])
    pred_df = pd.DataFrame(pred_list, columns=["emo_gt","emo_pred", "emo_pos", "img_path"])
    pred_df.to_csv(res_path_dir + "/" + f"{config.target_emo}_pred.csv", index=False)
    
    # calculate importance of features and save results
    importance_df = pd.DataFrame(gbm.feature_importance(), index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
    importance_df.to_csv(res_path_dir + "/" + f"{config.target_emo}_importance.csv")
    # visualize importance
    importance_df.plot(kind='bar', title=f'importance of {config.attribute} features')
    plt.savefig(res_path_dir + "/" + f"{config.target_emo}_importance.png")
    
    # save config
    res_path_rootdir = config.res_path_prefix + config.run_name
    config_dict = vars(config)
    with open(res_path_rootdir + '/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model configuration
    parser.add_argument('--run_name', type=str, default='default_run', help='run name')
    parser.add_argument('--other_run_name', type=str, default=None, help='run name of other test')
    parser.add_argument('--attribute', type=str, default='emotion', choices=['emotion', 'AU_sign', 'Gaze_sign', 'HP_sign'], help='attribute')
    parser.add_argument('--target_emo', type=str, default='comfort', choices=['comfort', 'discomfort'], help='target emotion')
    parser.add_argument('--num_leaves', type=int, default=31, help='number of leaves')
    parser.add_argument('--max_depth', type=int, default=-1, help='max depth')
    parser.add_argument('--select_features', type=str2bool, default=False, help='select features')
    
    # training configuration
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--num_boost_round', type=int, default=100, help='number of iterations')
    parser.add_argument('--early_stopping_rounds', type=int, default=10, help='early stopping rounds')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--is_class_weight', type=str2bool, default=False, help='is unbalance')
    parser.add_argument('--is_under_sampling', type=str2bool, default=False, help='is downsampling')
    parser.add_argument('--is_over_sampling', type=str2bool, default=False, help='is over sampling')
    
    # path configuration
    parser.add_argument('--model_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/Emotion_Estimator-snapshots/PIMD_A/', help='write path prefix')
    parser.add_argument('--res_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    parser.add_argument('--labels_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_labels_wsize30-ssize15.csv', help='labels directory')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_video_name_list_wsize30-ssize15.csv', help='video name list directory')
    parser.add_argument('--feats_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/JAANet_feature.pkl', help='au feats directory')
    
    config = parser.parse_args()
    
    main(config)