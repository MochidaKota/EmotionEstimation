import torch
import torch.nn as nn
import math

class EmoNet_1feature(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dims, dropout=None, batchnorm=False):
        super(EmoNet_1feature, self).__init__()
        
        layers = []
        
        if num_layers == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # define input layer
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            if batchnorm == True:
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(nn.ReLU(inplace=True))
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            
            # define hidden layers  
            for i in range(1, num_layers):
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                if batchnorm == True:
                    layers.append(nn.BatchNorm1d(hidden_dims[i]))
                layers.append(nn.ReLU(inplace=True))
                if dropout is not None:
                    layers.append(nn.Dropout(dropout))
            
            # define output layer       
            layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.emotion_output = nn.Sequential(*layers)
    
    def forward(self, x):
        emotion_output = self.emotion_output(x)
        return emotion_output

class EmoNet_2feature(nn.Module):
    def __init__(self, input_dims, output_dim, num_layers, hidden_dims, each_feat_dim=512, dropout=None, batchnorm=False, weighted=False, same_dim=False, summation=False):
        super(EmoNet_2feature, self).__init__()
        
        self.weighted = weighted
        self.same_dim = same_dim
        self.summation = summation
        self.each_feat_dim = each_feat_dim
        self.concat_dim = input_dims[0] + input_dims[1]
        layers = []
        
        if self.same_dim == True:
            self.embed_1st_feat = nn.Linear(input_dims[0], self.each_feat_dim)
            self.embed_2nd_feat = nn.Linear(input_dims[1], self.each_feat_dim)
            
        if self.weighted == True:
            if self.same_dim == True:
                self.weighted_layer = Weighted_Layer(self.each_feat_dim * 2, 2)
            else:
                self.weighted_layer = Weighted_Layer(self.concat_dim, 2)
        
        if num_layers == 0:
            if self.same_dim == True:
                if self.summation == True:
                    layers.append(nn.Linear(self.each_feat_dim, output_dim))
                else:
                    layers.append(nn.Linear(self.each_feat_dim * 2, output_dim))
            else:
                layers.append(nn.Linear(self.concat_dim, output_dim))
                
        else:
            # define input layer
            if self.same_dim == True:
                if self.summation == True:
                    layers.append(nn.Linear(self.each_feat_dim, hidden_dims[0]))
                else:
                    layers.append(nn.Linear(self.each_feat_dim * 2, hidden_dims[0]))
            else:
                layers.append(nn.Linear(self.concat_dim, hidden_dims[0]))
            if batchnorm == True:
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(nn.ReLU(inplace=True))
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        
            # define hidden layers      
            for i in range(1, num_layers):
                    layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                    if batchnorm == True:
                        layers.append(nn.BatchNorm1d(hidden_dims[i]))
                    layers.append(nn.ReLU(inplace=True))
                    if dropout is not None:
                        layers.append(nn.Dropout(dropout))
            
            # define output layer           
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.emotion_output = nn.Sequential(*layers)
        
    def forward(self, x_1st, x_2nd):
        if self.same_dim == True:
            x_1st = self.embed_1st_feat(x_1st)
            x_2nd = self.embed_2nd_feat(x_2nd)
        
        if self.weighted == True:
            weight = self.weighted_layer((x_1st, x_2nd))
            x_1st = x_1st * weight[:, 0].unsqueeze(1)
            x_2nd = x_2nd * weight[:, 1].unsqueeze(1)
            
        if self.summation == True:
            x = x_1st + x_2nd
        else:
            x = torch.cat((x_1st, x_2nd), dim=1)
        
        emotion_output = self.emotion_output(x)
        
        return emotion_output
    
class EmoNet_3feature(nn.Module):
    def __init__(self, input_dims, output_dim, num_layers, hidden_dims, each_feat_dim=512, dropout=None, batchnorm=False, weighted=False, same_dim=False, summation=False):
        super(EmoNet_3feature, self).__init__()
        
        self.weighted = weighted
        self.same_dim = same_dim
        self.summation = summation
        self.each_feat_dim = each_feat_dim
        self.concat_dim = input_dims[0] + input_dims[1] + input_dims[2]
        layers = []
        
        if self.same_dim == True:
            self.embed_1st_feat = nn.Linear(input_dims[0], self.each_feat_dim)
            self.embed_2nd_feat = nn.Linear(input_dims[1], self.each_feat_dim)
            self.embed_3rd_feat = nn.Linear(input_dims[2], self.each_feat_dim)
            
        if self.weighted == True:
            if self.same_dim == True:
                self.weighted_layer = Weighted_Layer(self.each_feat_dim * 3, 3)
            else:
                self.weighted_layer = Weighted_Layer(self.concat_dim, 3)
        
        if num_layers == 0:
            if self.same_dim == True:
                if self.summation == True:
                    layers.append(nn.Linear(self.each_feat_dim, output_dim))
                else:
                    layers.append(nn.Linear(self.each_feat_dim * 3, output_dim))
            else:
                layers.append(nn.Linear(self.concat_dim, output_dim))
                
        else:
            # define input layer
            if self.same_dim == True:
                if self.summation == True:
                    layers.append(nn.Linear(self.each_feat_dim, hidden_dims[0]))
                else:
                    layers.append(nn.Linear(self.each_feat_dim * 3, hidden_dims[0]))
            else:
                layers.append(nn.Linear(self.concat_dim, hidden_dims[0]))
            if batchnorm == True:
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(nn.ReLU(inplace=True))
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        
            # define hidden layers      
            for i in range(1, num_layers):
                    layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                    if batchnorm == True:
                        layers.append(nn.BatchNorm1d(hidden_dims[i]))
                    layers.append(nn.ReLU(inplace=True))
                    if dropout is not None:
                        layers.append(nn.Dropout(dropout))
            
            # define output layer           
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.emotion_output = nn.Sequential(*layers)
        
    def forward(self, x_1st, x_2nd, x_3rd):
        if self.same_dim == True:
            x_1st = self.embed_1st_feat(x_1st)
            x_2nd = self.embed_2nd_feat(x_2nd)
            x_3rd = self.embed_3rd_feat(x_3rd)
        
        if self.weighted == True:
            weight = self.weighted_layer((x_1st, x_2nd, x_3rd))
            x_1st = x_1st * weight[:, 0].unsqueeze(1)
            x_2nd = x_2nd * weight[:, 1].unsqueeze(1)
            x_3rd = x_3rd * weight[:, 2].unsqueeze(1)
            
        if self.summation == True:
            x = x_1st + x_2nd + x_3rd
        else:
            x = torch.cat((x_1st, x_2nd, x_3rd), dim=1)
        
        emotion_output = self.emotion_output(x)
        
        return emotion_output
        
class Weighted_Layer(nn.Module):
    def __init__(self, feat_dim, feat_num):
        super(Weighted_Layer, self).__init__()
        
        layer = []
        layer.append(nn.Linear(feat_dim, feat_dim // 2))
        layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Linear(feat_dim // 2, feat_num))
        layer.append(nn.Softmax(dim=1))
        
        self.weighted_layer = nn.Sequential(*layer)    

    def forward(self, feat_list:tuple):
        x = torch.cat(feat_list, dim=1)
        
        weights = self.weighted_layer(x)
        
        return weights
   
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
    def __init__(self, num_classes, input_dim, hidden_dims, activation='ReLU', dropout=0.1, batchnorm=True, summation=False, ew_product=False, arith_mean=False):
        super(MLPClassifier, self).__init__()
        
        layers = []
        self.summation = summation
        self.ew_product = ew_product
        self.arith_mean = arith_mean
        
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
        
    def forward(self, x:list):
        if type(x) != list:
            raise ValueError('x must be list.')

        feat_num = len(x)
        
        if feat_num == 1:
            x = x[0]
        else:
            if self.summation == True:
                x = torch.sum(torch.stack(x), dim=0)
                
                if self.arith_mean == True:
                    x = x / feat_num
                    
            elif self.ew_product == True:
                x = torch.prod(torch.stack(x), dim=0)
                
            else:
                x = torch.cat(x, dim=-1)
        
        mid_feat = None
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i == len(self.classifier) - 2:
                mid_feat = x
        
        return x, mid_feat
            
class AttentivePooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation='relu'):
        super(AttentivePooling, self).__init__()
        
        self.linear = nn.Linear(input_dim, hidden_dim, bias=False)
        
        self.attention_vector = nn.Parameter(torch.empty(hidden_dim, dtype=torch.float32))
        nn.init.normal_(self.attention_vector, mean=0.0, std=0.05)
        
        self.activation = None
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('activation must be tanh, sigmoid or relu.')

    def forward(self, inputs):
        
        # linear transformation
        transformed_inputs = self.linear(inputs)
        
        # activation
        transformed_inputs = self.activation(transformed_inputs)
        
        # attention weights
        attention_scores = torch.matmul(transformed_inputs, self.attention_vector)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_weights = attention_weights.unsqueeze(-1)
        
        # weighted sum
        weighted_sum = torch.sum(inputs * attention_weights, dim=1)
        
        return weighted_sum, attention_weights

class OneDCNNClassifier(nn.Module):
    def __init__(self, num_classes, in_channel, hid_channels, batchnorm=True, maxpool=True, kernel_size=3, stride=1, padding=1):
        super(OneDCNNClassifier, self).__init__()
        
        conv_layers = []
        
        # define input layer
        conv_layers.append(nn.Conv1d(in_channel, hid_channels[0], kernel_size=kernel_size, stride=stride, padding=padding))
        if batchnorm == True:
            conv_layers.append(nn.BatchNorm1d(hid_channels[0]))
        conv_layers.append(nn.ReLU(inplace=True))
        if maxpool == True:
            conv_layers.append(nn.MaxPool1d(kernel_size=2))
            
        # define hidden layers
        for i in range(1, len(hid_channels)):
            conv_layers.append(nn.Conv1d(hid_channels[i-1], hid_channels[i], kernel_size=kernel_size, stride=stride, padding=padding))
            if batchnorm == True:
                conv_layers.append(nn.BatchNorm1d(hid_channels[i]))
            conv_layers.append(nn.ReLU(inplace=True))
            if maxpool == True:
                conv_layers.append(nn.MaxPool1d(kernel_size=2))
                
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(hid_channels[-1], num_classes)
        
    def forward(self, x:list):
        if type(x) != list:
            raise ValueError('x must be list.')
        if len(x) == 1:
            x = x[0]
        else:
            x = torch.cat(x, dim=-1)
             
        x = self.conv_layers(x)
        x = self.gap(x)
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
        
    def forward(self, x:list):
        if type(x) != list:
            raise ValueError('x must be list.')
        if len(x) == 1:
            x = x[0]
        else:
            x = torch.cat(x, dim=-1)
            
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        
        return x 
    
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
    
class BottleNeckLayer(nn.Module):
    def __init__(self, input_dim, ratio=4, dropout=0.1, batchnorm=True):
        super(BottleNeckLayer, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, input_dim // ratio))
        if batchnorm:
            layers.append(nn.BatchNorm1d(input_dim // ratio))
        layers.append(nn.ReLU(inplace=True))
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(input_dim // ratio, input_dim))
        
        self.bottleneck_layer = nn.Sequential(*layers)
        
    def forward(self, x):
        for i, layer in enumerate(self.bottleneck_layer):
            x = layer(x)
        
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
        self.t_conv2 = nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2)
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