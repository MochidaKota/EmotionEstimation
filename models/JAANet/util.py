import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import math
import random

def decide_emo_pred(com_conf, com_pred, dis_conf, dis_pred, img_paths):
    emo_pred = []
    for i in range(len(com_conf)):
        if com_pred[i] == 0 and dis_pred[i] == 0:
            emo_pred.append(0)
        elif com_pred[i] == 1 and dis_pred[i] == 0:
            emo_pred.append(1)
        elif com_pred[i] == 0 and dis_pred[i] == 1:
            emo_pred.append(2)
        elif com_pred[i] == 1 and dis_pred[i] == 1:
            print(f"conflict: {img_paths[i]} (com_conf: {com_conf[i]}, dis_conf: {dis_conf[i]})")
            if com_conf[i] > dis_conf[i]:
                emo_pred.append(1)
            else:
                emo_pred.append(2)
    
    return emo_pred
           

def cb_cross_entropy_loss(num_par_cls, beta, device):
    weight = None
    if beta is not None:
        weight = (1. - beta) / (1. - torch.pow(beta, torch.tensor(num_par_cls)))
        weight = weight / torch.sum(weight) * len(num_par_cls)
        weight = weight.to(device)

    return nn.CrossEntropyLoss(weight=weight, reduction='sum')


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def extract(model, target, inputs):

    def forward_hook(module, inputs, outputs):
        # 順伝搬の出力を features というグローバル変数に記録する
        global features
        features = outputs.detach().clone()
    
    # コールバック関数を登録する
    target.register_forward_hook(forward_hook)
    
    with torch.no_grad():
        model(inputs)

    return features    
 
def str2bool(v):
    return v.lower() in ('true')

def tensor2img(img):
    img = img.data.numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0))+ 1) / 2.0 * 255.0
    return img.astype(np.uint8)

def save_img(img, name, path):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(path + name + '.png')
    return img

# calculate validation loss(eval)
def emo_estimation_evalv1(loader, region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, net, au_net_emo, patience, threshold_, use_gpu=True):
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    
    total_loss = 0
    batch_num = 0
    early_stopper = 0
    best_score_flag_ = 0
    for i, batch in enumerate(loader):
        batch_num += 1
        input, emotion, au = batch
        if use_gpu:
            input, emotion, au = input.cuda(), emotion.long().cuda(), au.float().cuda()

        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())
        local_au_out_feat = local_au_net(region_feat, output_aus_map)
        global_au_out_feat = global_au_feat(region_feat)
        concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat), 1)
        feat_output = net(concat_au_feat)
        emo_output_ = au_net_emo(feat_output)
        emo_output = F.softmax(emo_output_, dim=1)
        emo_output = emo_output[:, 1]
        
        
        if i == 0:
            all_output = emo_output.data.cpu().float()
            all_emotion = emotion.data.cpu().long()
            emo_output_forloss = emo_output_.data.cpu().float()
            
        else:
            all_output = torch.cat((all_output, emo_output.data.cpu().float()), 0)
            all_emotion = torch.cat((all_emotion, emotion.data.cpu().long()), 0)
            emo_output_forloss = torch.cat((emo_output_forloss, emo_output_.data.cpu().float()), 0)
    
    emo_val_loss = ce(emo_output_forloss, all_emotion.ravel())
    val_loss = emo_val_loss
    
    emotion_pred_prob = all_output.data.numpy()
    np.set_printoptions(threshold = np.inf)
    print(emotion_pred_prob)
    emotion_pred_prob = emotion_pred_prob.ravel()
    emotion_actual = all_emotion.data.numpy()
    emotion_actual = emotion_actual.ravel()
    #np.set_printoptions(threshold = np.inf)
    #print(emotion_actual)
    
    emotion_pred = np.zeros(emotion_pred_prob.shape)
    emotion_pred[emotion_pred_prob < threshold_] = 0
    emotion_pred[emotion_pred_prob >= threshold_] = 1

    curr_actual = emotion_actual
    curr_pred = emotion_pred
    
    #cl_report = classification_report(curr_actual, curr_pred)
    #print(cl_report)
    f1score = f1_score(curr_actual, curr_pred)
    precision = precision_score(curr_actual, curr_pred)
    recall = recall_score(curr_actual, curr_pred)
    acc = accuracy_score(curr_actual, curr_pred)

    return f1score, precision, recall, acc, val_loss

def au_classification_evalv1(loader, region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, net, au_net, patience, threshold_, use_gpu=True):
    mse = nn.MSELoss()
    
    total_loss = 0
    batch_num = 0
    early_stopper = 0
    best_score_flag_ = 0
    for i, batch in enumerate(loader):
        batch_num += 1
        input, emotion, au = batch
        if use_gpu:
            input, emotion, au = input.cuda(), emotion.long().cuda(), au.float().cuda()

        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())
        local_au_out_feat = local_au_net(region_feat, output_aus_map)
        global_au_out_feat = global_au_feat(region_feat)
        concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat), 1)
        feat_output = net(concat_au_feat)
        au_output = au_net(feat_output)
        au_output = au_output.view(au_output.size(0), 2, int(au_output.size(1)/2))
        au_output = F.log_softmax(au_output, dim=1)
        aus_output = (au_output[:,1,:]).exp()
        
        if i == 0:
            all_au = au.data.cpu().float()
            aus_output_forloss = aus_output.data.cpu().float()
            
        else:
            all_au = torch.cat((all_au, au.data.cpu().float()), 0)
            aus_output_forloss = torch.cat((aus_output_forloss, aus_output.data.cpu().float()), 0)
    
    au_val_loss = mse(aus_output_forloss, all_au)
    val_loss = au_val_loss

    return val_loss



#predict emotion(eval, test)     
def emo_estimation_testv1(loader, region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, net, au_net_emo, threshold, use_gpu=True):
    for i, batch in enumerate(loader):
        input, emotion = batch
        if use_gpu:
            input, emotion = input.cuda(), emotion.cuda()

        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())
        local_au_out_feat = local_au_net(region_feat, output_aus_map)
        global_au_out_feat = global_au_feat(region_feat)
        concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat), 1)
        feat_output = net(concat_au_feat)
        emo_output = au_net_emo(feat_output)
        emo_output = F.softmax(emo_output, dim=1)
        emo_output = emo_output[:, 1]
        
        if i == 0:
            all_output = emo_output.data.cpu().float()
            all_emotion = emotion.data.cpu().long()
        else:
            all_output = torch.cat((all_output, emo_output.data.cpu().float()), 0)
            all_emotion = torch.cat((all_emotion, emotion.data.cpu().long()), 0)
    
    emotion_pred_prob = all_output.data.numpy()
    emotion_pred_prob = emotion_pred_prob.ravel()
    emotion_actual = all_emotion.data.numpy()
    emotion_actual = emotion_actual.ravel()
    
    emotion_pred = np.zeros(emotion_pred_prob.shape)
    emotion_pred[emotion_pred_prob <= threshold] = 0
    emotion_pred[emotion_pred_prob > threshold] = 1

    curr_actual = emotion_actual
    curr_pred = emotion_pred
    
    '''
    errors_fn = np.zeros(len(curr_actual))
    errors_fp = np.zeros(len(curr_actual))
    for i in range(len(curr_actual)):
        if curr_actual[i] == 1 and curr_pred[i] == 0:
            errors_fn[i] = 1
        if curr_actual[i] == 0 and curr_pred[i] == 1:
            errors_fp[i] = 1
    '''
    
    #np.set_printoptions(threshold = np.inf)
    #cm = confusion_matrix(curr_actual, curr_pred)
    #print(cm)
    #cl_report = classification_report(curr_actual, curr_pred)
    #print(cl_report)
    f1score = f1_score(curr_actual, curr_pred)
    precision = precision_score(curr_actual, curr_pred)
    recall = recall_score(curr_actual, curr_pred)
    acc = accuracy_score(curr_actual, curr_pred)
    queue_rate = np.mean(curr_pred)

    return f1score, precision, recall, acc, queue_rate

#calculate AU probs(test)
def AU_detection_testv1(loader, region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, net, au_net, use_gpu=True):
    for i, batch in enumerate(loader):
        input, emotion = batch
        if use_gpu:
            input, emotion = input.cuda(), emotion.cuda()

        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())
        local_au_out_feat = local_au_net(region_feat, output_aus_map)
        global_au_out_feat = global_au_feat(region_feat)
        concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat), 1)
        feat_output = net(concat_au_feat)
        au_output = au_net(feat_output)
        au_output = au_output.view(au_output.size(0), 2, int(au_output.size(1)/2))
        au_output = F.log_softmax(au_output, dim=1)
        aus_output = (au_output[:,1,:]).exp()
        
        if i == 0:
            all_output = aus_output.data.cpu().float()
            all_emotion = emotion.data.cpu().long()
        else:
            all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
            all_emotion = torch.cat((all_emotion, emotion.data.cpu().long()), 0)
    
    all_output = all_output.detach().numpy().copy()
    all_emotion = all_emotion.detach().numpy().copy()
    
    columns = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
    AUprobs = pd.DataFrame(data=all_output, columns=columns, dtype='float')
    emotion = pd.DataFrame(data=all_emotion, columns=['emotion'], dtype='long')
    AU_probs = pd.concat([AUprobs, emotion], axis=1)
    return AU_probs


def AU_detection_evalv2(loader, region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, au_net, use_gpu=True, fail_threshold = 0.1):
    missing_label = 9
    for i, batch in enumerate(loader):
        input, land, biocular, au  = batch
        if use_gpu:
            input, land, au = input.cuda(), land.cuda(), au.cuda()

        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())
        local_au_out_feat, local_aus_output = local_au_net(region_feat, output_aus_map)
        local_aus_output = (local_aus_output[:, 1, :]).exp()
        global_au_out_feat = global_au_feat(region_feat)
        concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat.detach()), 1)
        aus_output = au_net(concat_au_feat)
        aus_output = (aus_output[:,1,:]).exp()

        if i == 0:
            all_local_output = local_aus_output.data.cpu().float()
            all_output = aus_output.data.cpu().float()
            all_au = au.data.cpu().float()
            all_pred_land = align_output.data.cpu().float()
            all_land = land.data.cpu().float()
        else:
            all_local_output = torch.cat((all_local_output, local_aus_output.data.cpu().float()), 0)
            all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
            all_au = torch.cat((all_au, au.data.cpu().float()), 0)
            all_pred_land = torch.cat((all_pred_land, align_output.data.cpu().float()), 0)
            all_land = torch.cat((all_land, land.data.cpu().float()), 0)

    AUoccur_pred_prob = all_output.data.numpy()
    local_AUoccur_pred_prob = all_local_output.data.numpy()
    AUoccur_actual = all_au.data.numpy()
    pred_land = all_pred_land.data.numpy()
    GT_land = all_land.data.numpy()
    # np.savetxt('BP4D_part1_pred_land_49.txt', pred_land, fmt='%.4f', delimiter='\t')
    np.savetxt('B3D_val_predAUprob-2_all_.txt', AUoccur_pred_prob, fmt='%f',
               delimiter='\t')
    # AUs
    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1
    local_AUoccur_pred = np.zeros(local_AUoccur_pred_prob.shape)
    local_AUoccur_pred[local_AUoccur_pred_prob < 0.5] = 0
    local_AUoccur_pred[local_AUoccur_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))
    local_AUoccur_pred = local_AUoccur_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    local_f1score_arr = np.zeros(AUoccur_actual.shape[0])
    local_acc_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]
        local_curr_pred = local_AUoccur_pred[i]

        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]
        local_new_curr_pred = local_curr_pred[curr_actual != missing_label]

        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)
        local_f1score_arr[i] = f1_score(new_curr_actual, local_new_curr_pred)
        local_acc_arr[i] = accuracy_score(new_curr_actual, local_new_curr_pred)

    # landmarks
    errors = np.zeros((GT_land.shape[0], int(GT_land.shape[1] / 2)))
    mean_errors = np.zeros(GT_land.shape[0])
    for i in range(GT_land.shape[0]):
        left_eye_x = GT_land[i, (20 - 1) * 2:(26 - 1) * 2:2]
        l_ocular_x = left_eye_x.mean()
        left_eye_y = GT_land[i, (20 - 1) * 2 + 1:(26 - 1) * 2 + 1:2]
        l_ocular_y = left_eye_y.mean()

        right_eye_x = GT_land[i, (26 - 1) * 2:(32 - 1) * 2:2]
        r_ocular_x = right_eye_x.mean()
        right_eye_y = GT_land[i, (26 - 1) * 2 + 1:(32 - 1) * 2 + 1:2]
        r_ocular_y = right_eye_y.mean()

        biocular = math.sqrt((l_ocular_x - r_ocular_x) ** 2 + (l_ocular_y - r_ocular_y) ** 2)

        for j in range(0, GT_land.shape[1], 2):
            errors[i, int(j / 2)] = math.sqrt((GT_land[i, j] - pred_land[i, j]) ** 2 + (
                    GT_land[i, j + 1] - pred_land[i, j + 1]) ** 2) / biocular

        mean_errors[i] = errors[i].mean()
    mean_error = mean_errors.mean()

    failure_ind = np.zeros(len(GT_land))
    failure_ind[mean_errors > fail_threshold] = 1
    failure_rate = failure_ind.sum() / failure_ind.shape[0]

    return local_f1score_arr, local_acc_arr, f1score_arr, acc_arr, mean_error, failure_rate


def vis_attention(loader, region_learning, align_net, local_attention_refine, write_path_prefix, net_name, epoch, alpha = 0.5, use_gpu=True):
    for i, batch in enumerate(loader):
        input, land, biocular, au = batch
        # if i > 1:
        #     break
        if use_gpu:
            input = input.cuda()
        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())

        # aus_map is predefined, and output_aus_map is refined
        spatial_attention = output_aus_map #aus_map
        if i == 0:
            all_input = input.data.cpu().float()
            all_spatial_attention = spatial_attention.data.cpu().float()
        else:
            all_input = torch.cat((all_input, input.data.cpu().float()), 0)
            all_spatial_attention = torch.cat((all_spatial_attention, spatial_attention.data.cpu().float()), 0)

    for i in range(all_spatial_attention.shape[0]):
        background = save_img(all_input[i], 'input', write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_')
        for j in range(all_spatial_attention.shape[1]):
            fig, ax = plt.subplots()
            # print(all_spatial_attention[i,j].max(), all_spatial_attention[i,j].min())
            # cax = ax.imshow(all_spatial_attention[i,j], cmap='jet', interpolation='bicubic')
            cax = ax.imshow(all_spatial_attention[i, j], cmap='jet', interpolation='bicubic', vmin=0, vmax=1)
            ax.axis('off')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            #        cbar = fig.colorbar(cax)
            fig.savefig(write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

        for j in range(all_spatial_attention.shape[1]):
            overlay = Image.open(write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png')
            overlay = overlay.resize(background.size, Image.ANTIALIAS)
            background = background.convert('RGBA')
            overlay = overlay.convert('RGBA')
            new_img = Image.blend(background, overlay, alpha)
            new_img.save(write_path_prefix + net_name + '/overlay_vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png', 'PNG')


def dice_loss(pred, target, smooth = 1):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth)) / iflat.size(0)


def au_softmax_loss(input, target, weight=None, size_average=True, reduce=True):
    classify_loss = nn.NLLLoss(size_average=size_average, reduce=reduce)

    for i in range(input.size(2)):
        t_input = input[:, :, i]
        t_target = target[:, i]
        t_loss = classify_loss(t_input, t_target)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def au_dice_loss(input, target, weight=None, smooth = 1, size_average=True):
    for i in range(input.size(2)):
        # input is log_softmax, t_input is probability
        t_input = (input[:, 1, i]).exp()
        t_target = (target[:, i]).float()
        # t_loss = 1 - float(2*torch.dot(t_input, t_target) + smooth)/\
        #          (torch.dot(t_input, t_input)+torch.dot(t_target, t_target)+smooth)/t_input.size(0)
        t_loss = dice_loss(t_input, t_target, smooth)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def landmark_loss(input, target, biocular, size_average=True):
    for i in range(input.size(0)):
        t_input = input[i,:]
        t_target = target[i,:]
        t_loss = torch.sum((t_input - t_target) ** 2) / (2.0*biocular[i])
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def attention_refine_loss(input, target, size_average=True, reduce=True):
    # loss is averaged over each point in the attention map,
    # note that Eq.(4) in our ECCV paper is to sum all the points,
    # change the value of lambda_refine can remove this difference.
    classify_loss = nn.BCELoss(size_average=size_average, reduce=reduce)

    input = input.view(input.size(0), input.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    for i in range(input.size(1)):
        t_input = input[:, i, :]
        t_target = target[:, i, :]
        t_loss = classify_loss(t_input, t_target)
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)
    # sum losses of all AUs
    return loss.sum()
