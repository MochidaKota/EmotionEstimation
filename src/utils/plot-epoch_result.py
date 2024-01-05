import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse

def plot_metrics(config):
    metrics_list = pd.DataFrame()
    for i, metric_path in enumerate(glob.glob(config.res_path_prefix + config.run_name + '/**/' + 'metrics.csv', recursive=True)):
        _metrics = pd.read_csv(metric_path)
        _metrics['epoch'] = i + 1
        metrics_list = pd.concat([metrics_list, _metrics], axis=0)

    metrics_list = metrics_list.reset_index(drop=True)

    fig = plt.figure()
    plt.plot(metrics_list['epoch'], metrics_list['precision'], label='precision')
    plt.plot(metrics_list['epoch'], metrics_list['recall'], label='recall')
    plt.plot(metrics_list['epoch'], metrics_list['f1'], label='f1')
    plt.plot(metrics_list['epoch'], metrics_list['accuracy'], label='accuracy')
    plt.plot(metrics_list['epoch'], metrics_list['roc_auc'], label='roc_auc')
    plt.plot(metrics_list['epoch'], metrics_list['pr_auc'], label='pr_auc')
    plt.xlabel('epoch')
    plt.ylabel('metrics')
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.res_path_prefix + config.run_name + '/metrics.png')
    
    print()
    print(f'-----plot metrics-----')
    print()
    
def plot_history(config):
    fig = plt.figure(figsize=(12, 6))

    for i, history_path in enumerate(glob.glob(config.res_path_prefix + config.run_name + '/history/**/' + '*_history.csv', recursive=True)):
        _history = pd.read_csv(history_path)
        plt.subplot(1, 2, 1)
        plt.plot(_history['epoch'], _history['train_loss'], label=f'fold{i+1}')
        plt.subplot(1, 2, 2)
        plt.plot(_history['epoch'], _history['test_loss'], label=f'fold{i+1}')

    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('test_loss')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(config.res_path_prefix + config.run_name + '/history.png')
    
    print()
    print(f'-----plot history-----')
    print()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--res_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/reports/PIMD_A/', help='write result prefix')
    config = parser.parse_args()
    
    plot_metrics(config)
    
    plot_history(config)