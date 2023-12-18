import math
import time

import torch
from torch import nn

from sixDRepNet_backbone.repvgg import get_RepVGG_func_by_name
from utils import sixdrepnet_util

class SixDRepNet(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 pretrained=True):
        super(SixDRepNet, self).__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 6)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        hp_feat = x
        x = self.linear_reg(x)
        hp_logits = x
        
        return sixdrepnet_util.compute_rotation_matrix_from_ortho6d(x), hp_feat, hp_logits
    
def get_seq_pooling_from_sixdrepnet(sixdrepnet, x, seq_len):
    # x: (batch_size, seq_len, c, h, w)
    # input x to sixdrepnet to get the feature vector
    # then pooling the feature vector
    # pooling type: mean, max
    
    batch_size = x.shape[0]
    c = x.shape[2]
    h = x.shape[3]
    w = x.shape[4]
    x = x.reshape(batch_size*seq_len, c, h, w)
    _, hp_feat = sixdrepnet(x)
    hp_feat = hp_feat.reshape(batch_size, seq_len, -1)
    hp_feat_mean = torch.mean(hp_feat, dim=1)
    hp_feat_max = torch.max(hp_feat, dim=1)[0]
    
    return hp_feat_mean, hp_feat_max    