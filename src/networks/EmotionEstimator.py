import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Create a matrix of shape (max_seq_len, d_model) for positional encodings
        pe = torch.zeros(max_seq_len, d_model)
        
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return math.sqrt(x.size(-1)) * x + self.pe[:, :x.size(1)]
  
class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, d_model, num_heads, d_hid, num_layers, max_seq_len, dropout=0.1, wo_pe=False, pool_type='mean'):
        super(TransformerClassifier, self).__init__()
        
        self.pool_type = pool_type
        self.max_seq_len = max_seq_len
        _d_model = d_model
        if self.pool_type not in ['mean', 'max', 'cls', 'mean+max', 'att']:
            raise ValueError('pool_type must be mean, max, cls, mean+max or att.')
        if self.pool_type == 'cls':
            self.max_seq_len += 1
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        if pool_type == 'mean+max':
            _d_model = d_model * 2
        if pool_type == 'att':
            self.att_pool = AttentivePooling(input_dim=d_model, hidden_dim=d_model // 2)
        
        self.embedding = nn.Linear(input_dim, d_model, bias=False)
        nn.init.xavier_uniform_(self.embedding.weight)
            
        self.wo_pe = wo_pe
        if self.wo_pe == False:
            self.pe = PositionalEncoding(d_model=d_model, max_seq_len=self.max_seq_len)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_hid, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(_d_model, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x:list):
        if len(x) == 1:
            x = x[0]
        else:
            x = torch.cat(x, dim=-1)

        x = self.embedding(x)
        
        if self.pool_type == 'cls':
            # CLSトークンを追加
            cls_token = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        
        if self.wo_pe == False:
            x = self.pe(x)
            
        x = self.encoder(x)
        
        # 時間方向にプーリング or CLSトークンを取得して特徴ベクトルとする
        if self.pool_type == 'mean':
            x = x.mean(dim=1)
        elif self.pool_type == 'max':
            x = x.max(dim=1)[0]
        elif self.pool_type == 'mean+max':
            x = torch.cat((x.mean(dim=1), x.max(dim=1)[0]), dim=1)
        elif self.pool_type == 'cls':
            x = x[:, 0, :]
        elif self.pool_type == 'att':
            x = self.att_pool(x)
    
        mid_feat = x
        x = self.fc(x)
        
        return x, mid_feat
    
class MLPClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dims, activation='ReLU', dropout=0.1, batchnorm=True):
        super(MLPClassifier, self).__init__()
        
        layers = []
        
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('activation must be ReLU or LeakyReLU.')
        
        if len(hidden_dims) == 0:
            layers.append(nn.Linear(input_dim, num_classes))
        else:
            # define input layer
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            if batchnorm == True:
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(self.activation)
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        
            # define hidden layers  
            for i in range(1, len(hidden_dims)):
                    layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                    if batchnorm == True:
                        layers.append(nn.BatchNorm1d(hidden_dims[i]))
                    layers.append(self.activation)
                    if dropout is not None:
                        layers.append(nn.Dropout(dropout))
        
            # define output layer           
            layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        
        mid_feat = None
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i == len(self.classifier) - 2:
                mid_feat = x
        
        return x, mid_feat
            
class AttentivePooling(nn.Module):
    def __init__(self, input_dim, pool_type='base'):
        super(AttentivePooling, self).__init__()
        
        self.pool_type = pool_type
        self.attention_vector = None
        self.linear = None
        
        if self.pool_type == 'base':
            self.attention_vector = nn.Parameter(torch.empty(input_dim, dtype=torch.float32))
            nn.init.normal_(self.attention_vector, mean=0.0, std=0.05)
            
        elif self.pool_type == 'woLi':
            self.linear = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(inplace=True)
            )
            self.attention_vector = nn.Parameter(torch.empty(input_dim // 2, dtype=torch.float32))
            nn.init.normal_(self.attention_vector, mean=0.0, std=0.05)
            
        elif self.pool_type == 'Li':
            self.linear = nn.Linear(input_dim, 1, bias=False)
            
        elif self.pool_type == 'MLP':
            self.linear = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim // 2, 1)
            )

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, input_dim)
        
        # calculate attention scores
        if self.pool_type == 'base':
            attention_scores = torch.matmul(inputs, self.attention_vector.unsqueeze(-1)).squeeze(-1)
            
        elif self.pool_type == 'woLi':
            transformed_inputs = self.linear(inputs)
            attention_scores = torch.matmul(transformed_inputs, self.attention_vector.unsqueeze(-1)).squeeze(-1)
            
        elif self.pool_type == 'Li' or self.pool_type == 'MLP':
            attention_scores = self.linear(inputs).view(inputs.size(0), -1)
        
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # weighted sum
        weighted_sum = torch.sum(inputs * attention_weights.unsqueeze(-1), dim=1)
        
        return weighted_sum, attention_weights

class Conv1DClassifier(nn.Module):
    def __init__(self, num_classes, in_channel, hid_channels, batchnorm=True, maxpool=False, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], gpool_type='avg', att_pool_type='base'):
        super(Conv1DClassifier, self).__init__()
        
        conv_layers = []
        
        # define input layer
        conv_layers.append(nn.Conv1d(in_channel, hid_channels[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0]))
        if batchnorm == True:
            conv_layers.append(nn.BatchNorm1d(hid_channels[0]))
        conv_layers.append(nn.ReLU(inplace=True))
        if maxpool == True and len(hid_channels) > 1:
            conv_layers.append(nn.MaxPool1d(kernel_size=2))
            
        # define hidden layers
        for i in range(1, len(hid_channels)):
            conv_layers.append(nn.Conv1d(hid_channels[i-1], hid_channels[i], kernel_size=kernel_size[i], stride=stride[i], padding=padding[i]))
            if batchnorm == True:
                conv_layers.append(nn.BatchNorm1d(hid_channels[i]))
            conv_layers.append(nn.ReLU(inplace=True))
            if maxpool == True and i != len(hid_channels) - 1:
                conv_layers.append(nn.MaxPool1d(kernel_size=2))
                
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.global_pooling = None
        self.gpool_type = gpool_type
        if gpool_type == 'avg':
            self.global_pooling = nn.AdaptiveAvgPool1d(1)
        elif gpool_type == 'max':   
            self.global_pooling = nn.AdaptiveMaxPool1d(1)
        elif gpool_type == 'att':
            self.global_pooling = AttentivePooling(input_dim=hid_channels[-1], pool_type=att_pool_type)
        
        self.fc = nn.Linear(hid_channels[-1], num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len, in_channel)
             
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        
        if self.gpool_type == 'att':
            x = x.transpose(1, 2)
            x, _ = self.global_pooling(x)
        else:
            x = self.global_pooling(x)
            
        x = x.view(x.size(0), -1)
        mid_feat = x
        
        x = self.fc(x)
        
        return x, mid_feat
    
class Conv2DClassifier(nn.Module):
    def __init__(self, num_classes, hid_channels, in_channel=1, batchnorm=True, maxpool=True, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], gpool_type='avg'):
        super(Conv2DClassifier, self).__init__()
        
        conv_layers = []
        
        # define input layer
        conv_layers.append(nn.Conv2d(in_channel, hid_channels[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0]))
        if batchnorm == True:
            conv_layers.append(nn.BatchNorm2d(hid_channels[0]))
        conv_layers.append(nn.ReLU(inplace=True))
        if maxpool == True and len(hid_channels) > 1:
            conv_layers.append(nn.MaxPool2d(kernel_size=2))
            
        # define hidden layers
        for i in range(1, len(hid_channels)):
            conv_layers.append(nn.Conv2d(hid_channels[i-1], hid_channels[i], kernel_size=kernel_size[i], stride=stride[i], padding=padding[i]))
            if batchnorm == True:
                conv_layers.append(nn.BatchNorm2d(hid_channels[i]))
            conv_layers.append(nn.ReLU(inplace=True))
            if maxpool == True and i != len(hid_channels) - 1:
                conv_layers.append(nn.MaxPool2d(kernel_size=2))
                
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.global_pooling = None
        self.gpool_type = gpool_type
        if gpool_type == 'avg':
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        elif gpool_type == 'max':   
            self.global_pooling = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Linear(hid_channels[-1], num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        
        x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        
        x = self.global_pooling(x)
        
        x = x.view(x.size(0), -1)
        mid_feat = x
        
        x = self.fc(x)
        
        return x, mid_feat
    
class LSTMClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, num_layers, dropout=0, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        if bidirectional == True:
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
        else:
            self.fc = nn.Linear(hidden_dim, num_classes)
        
        # initialize weights
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
            
        x, _ = self.lstm(x)
        
        # x = x[:, -1, :]
        
        mid_feat = x
        
        x = self.fc(x)
        
        return x , mid_feat
    
class StreamMixer(nn.Module):
    def __init__(self, feats_dim, num_feats, hidden_dims, dropout=0.1, activation='relu', batchnorm=True, is_binary=False):
        super(StreamMixer, self).__init__()
        
        self.is_binary = is_binary
        
        layers = []
        if activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'relu':
            activation = nn.ReLU(inplace=True)
        else:
            raise ValueError('activation must be tanh or relu.')
        
        if len(hidden_dims) == 0:
            layers.append(nn.Linear(feats_dim * num_feats, num_feats))
            layers.append(nn.Softmax(dim=1))
            
        else:
            # define input layer
            layers.append(nn.Linear(feats_dim * num_feats, hidden_dims[0]))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(activation)
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            
            # define hidden layers
            for i in range(1, len(hidden_dims)):
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                if batchnorm:
                    layers.append(nn.BatchNorm1d(hidden_dims[i]))
                layers.append(activation)
                if dropout is not None:
                    layers.append(nn.Dropout(dropout))
            
            # define output layer
            layers.append(nn.Linear(hidden_dims[-1], num_feats))
            layers.append(nn.Softmax(dim=1))
        
        self.attention_weight = nn.Sequential(*layers)
        
    def binary_attention(self, x):
        # x: (batch_size, attention_weights)
        # 各batchに対して、attention_weightsの値が最大のものを1、それ以外を0とする
        max_index = torch.argmax(x, dim=1)
        binary_attention = torch.zeros_like(x)
        binary_attention[torch.arange(x.size(0)), max_index] = 1
        
        return binary_attention
                
    def forward(self, x:list):
        if type(x) != list:
            raise ValueError('x must be list.')
        
        x = torch.cat(x, dim=-1)
        
        for i, layer in enumerate(self.attention_weight):
            x = layer(x)
            
        if self.is_binary == True:
            x = self.binary_attention(x)
            
        return x
    
class AUStream(nn.Module):
    def __init__(self, num_classes=2, input_dim=12000, hidden_dim=128, hid_channels=[256, 512], dropout=0.1, kernel_size=5, is_maxpool=True):
        super(AUStream, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        self.conv1 = nn.Conv1d(hidden_dim, hid_channels[0], kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(hid_channels[0])
        self.conv2 = nn.Conv1d(hid_channels[0], hid_channels[1], kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm1d(hid_channels[1])
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.fc2 = nn.Linear(hid_channels[1], num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.is_maxpool = is_maxpool
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        
    def forward(self, x):
        x = self.fc1(x)
        
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.is_maxpool:
            x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
    
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        mid_feat = x
        
        x = self.fc2(x)
        
        return x, mid_feat
    
class AUStream2(nn.Module):
    def __init__(self, num_classes=2, window_size=30, is_maxpool=False, global_pooling='ave'):
        super(AUStream2, self).__init__()
        
        kernel_size = 20
        
        self.s_conv1 = nn.Conv1d(window_size, window_size, kernel_size=kernel_size, stride=kernel_size // 2, padding=6)
        self.s_bn1 = nn.BatchNorm1d(window_size)
        self.s_conv2 = nn.Conv1d(window_size, window_size, kernel_size=kernel_size, stride=kernel_size // 2, padding=6)
        self.s_bn2 = nn.BatchNorm1d(window_size)
        
        self.t_conv1 = nn.Conv1d(120, 256, kernel_size=3, stride=1, padding=1)
        self.t_bn1 = nn.BatchNorm1d(256)
        self.t_conv2 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.t_bn2 = nn.BatchNorm1d(512)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.global_pooling = None
        self.gpool_type = global_pooling
        if global_pooling == 'ave':
            self.global_pooling = nn.AdaptiveAvgPool1d(1)
        elif global_pooling == 'max':
            self.global_pooling = nn.AdaptiveMaxPool1d(1)
        elif global_pooling == 'att':
            self.global_pooling = AttentivePooling(input_dim=512, hidden_dim=128)
        
        self.fc = nn.Linear(512, num_classes)
        
        if is_maxpool:
            self.maxpool = nn.MaxPool1d(kernel_size=3)
        
    def forward(self, x):
        x = self.s_conv1(x)
        x = self.s_bn1(x)
        x = self.relu(x)
    
        x = self.s_conv2(x)
        x = self.s_bn2(x)
        x = self.relu(x)
        
        x = x.transpose(1, 2)
        
        x = self.t_conv1(x)
        x = self.t_bn1(x)
        x = self.relu(x)
        
        x = self.t_conv2(x)
        x = self.t_bn2(x)
        x = self.relu(x)
        
        if self.gpool_type == 'att':
            x = x.transpose(1, 2)
        
        x = self.global_pooling(x)
        
        if self.gpool_type == 'att':
            x, _ = x
        
        x = x.view(x.size(0), -1)
        mid_feat = x
        
        x = self.fc(x)
        
        return x, mid_feat
    
class AUStreamTemp(nn.Module):
    def __init__(self, num_classes=2, input_dim=12000, hid_channels=[256, 512], global_pooling='ave'):
        super(AUStreamTemp, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, hid_channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hid_channels[0])
        self.conv2 = nn.Conv1d(hid_channels[0], hid_channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(hid_channels[1])
        
        self.relu = nn.ReLU(inplace=True)
        
        self.global_pooling = None
        self.gpool_type = global_pooling
        if global_pooling == 'ave':
            self.global_pooling = nn.AdaptiveAvgPool1d(1)
        elif global_pooling == 'max':
            self.global_pooling = nn.AdaptiveMaxPool1d(1)
        elif global_pooling == 'att':
            self.global_pooling = AttentivePooling(input_dim=512, hidden_dim=128)
            
        self.fc = nn.Linear(hid_channels[1], num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        if self.gpool_type == 'att':
            x = x.transpose(1, 2)
        
        x = self.global_pooling(x)
        
        if self.gpool_type == 'att':
            x, _ = x
        
        x = x.view(x.size(0), -1)
        mid_feat = x
        
        x = self.fc(x)
        
        return x, mid_feat
    
class GazeStream(nn.Module):
    def __init__(self, num_classes=2, input_dim=180, hid_channels=[256, 512], global_pooling='ave'):
        super(GazeStream, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, hid_channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hid_channels[0])
        self.conv2 = nn.Conv1d(hid_channels[0], hid_channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(hid_channels[1])
        
        self.relu = nn.ReLU(inplace=True)
        
        self.global_pooling = None
        self.gpool_type = global_pooling
        if global_pooling == 'ave':
            self.global_pooling = nn.AdaptiveAvgPool1d(1)
        elif global_pooling == 'max':
            self.global_pooling = nn.AdaptiveMaxPool1d(1)
        elif global_pooling == 'att':
            self.global_pooling = AttentivePooling(input_dim=512, hidden_dim=128)
            
        self.fc = nn.Linear(hid_channels[1], num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        if self.gpool_type == 'att':
            x = x.transpose(1, 2)
        
        x = self.global_pooling(x)
        
        if self.gpool_type == 'att':
            x, _ = x
        
        x = x.view(x.size(0), -1)
        mid_feat = x
        
        x = self.fc(x)
        
        return x, mid_feat
    
    
class HPStream(nn.Module):
    def __init__(self, num_classes=2, input_dim=6, hid_channels=[8, 512], global_pooling='avg'):
        super(HPStream, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, hid_channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hid_channels[0])
        self.conv2 = nn.Conv1d(hid_channels[0], hid_channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(hid_channels[1])
        
        self.relu = nn.ReLU(inplace=True)
        
        self.global_pooling = None
        self.gpool_type = global_pooling
        if global_pooling == 'avg':
            self.global_pooling = nn.AdaptiveAvgPool1d(1)
        elif global_pooling == 'max':
            self.global_pooling = nn.AdaptiveMaxPool1d(1)
        elif global_pooling == 'att':
            self.global_pooling = AttentivePooling(input_dim=512, hidden_dim=128)
            
        self.fc = nn.Linear(hid_channels[1], num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        if self.gpool_type == 'att':
            x = x.transpose(1, 2)
        
        x = self.global_pooling(x)
        
        if self.gpool_type == 'att':
            x, _ = x
        
        x = x.view(x.size(0), -1)
        mid_feat = x
        
        x = self.fc(x)
        
        return x, mid_feat