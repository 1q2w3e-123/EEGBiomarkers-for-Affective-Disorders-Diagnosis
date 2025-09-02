import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from copy import deepcopy
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
import torch.nn.init as init
# from nimh.transformer2 import Conformer
########################################################################################


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()
        
    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x
        
        
class MRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size,input=1):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.GELU = nn.GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(input, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(input, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        # print(x_concat.shape)
        return x_concat
    
def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        nbatches = query.size(0)

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key   = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linear(x)


class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TCE(nn.Module):
    '''
    Transformer Encoder

    It is a stack of N layers.
    '''

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EncoderLayer(nn.Module):
    '''
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''
    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)


    def forward(self, x_in):
        "Transformer Encoder"
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))  # Encoder self-attention
        return self.sublayer_output[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Positionwise feed-forward network."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class AttnSleep(nn.Module):
    def __init__(self,input=1):
        super(AttnSleep, self).__init__()

        N = 4  # number of TCE clones
        d_model = 80  # set to be 100 for SHHS dataset
        d_ff = 120   # dimension of feed forward
        h = 5  # number of attention heads
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30

        self.mrcnn = MRCNN(afr_reduced_cnn_size,input=input) # use MRCNN_SHHS for SHHS dataset

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)

    def forward(self, x):

        x_feat = self.mrcnn(x)

        encoded_features = self.tce(x_feat)
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        final_output = self.fc(encoded_features)
        return final_output
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.downsample=nn.Conv1d(60,20,kernel_size=11,stride=1,padding=5)
        self.downnorm=nn.BatchNorm1d(20)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            residual = self.downnorm(residual)


        out += residual
        out = self.relu(out)

        return out
    
class SE_GRU(nn.Module):
    def __init__(self, input=1,cnn_hidden_size=20, rnn_hidden_size=100, num_layers=2, num_classes=3):
        super(SE_GRU, self).__init__()

        # CNN部分
        self.conv1 = nn.Conv1d(in_channels=input, out_channels=cnn_hidden_size, kernel_size=(50,), stride=(1,))
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=(10,), stride=(5,))
        self.se_block=SEBasicBlock(60,20)
        # RNN部分
        self.gru = nn.GRU(cnn_hidden_size, rnn_hidden_size, num_layers, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        # CNN部分
        # print(x.shape)
        # x = self.conv1(x)
        # x = self.relu(x)
        # cnn_output = self.pool1(x)
        cnn_output=self.se_block(x)
        # print(cnn_output.shape)
        # 将CNN的输出维度转换为RNN的输入维度
        rnn_input = cnn_output.permute(0, 2, 1)
        # rnn_input = x.permute(0, 2, 1)

        # RNN部分
        rnn_out = self.gru(rnn_input)[0]
        # 取LSTM层的最后一个时间步的输出作为模型输出
        output = self.fc(rnn_out[:, -1, :])

        return output

class TimeTransformer(nn.Module):
    # def __init__(self, input_len=128, dim=64, heads=4, depth=2):
    def __init__(self, input_channels=40, seq_len=128, num_classes=2, embed_dim=64, num_heads=4, ff_dim=128):
        super().__init__()
        # self.embed = nn.Linear(input_len, dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # shape: [B, embed_dim, 1]
            nn.Flatten(),             # shape: [B, embed_dim]
            # nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x):
        # print(x.shape)
        # assert(1==0)
        # x: [B, C, 128]
        x = self.cnn(x)  # [B, embed_dim, 128]
        x = x.permute(0, 2, 1)  # [B, 128, embed_dim]
        x = x + self.pos_embedding[:, :x.size(1), :]  # Add positional encoding
        x = self.transformer(x)  # [B, 128, embed_dim]
        x = x.permute(0, 2, 1)  # [B, embed_dim, 128]
        # print(x.shape)
        # assert(1==0)
        out = self.classifier(x)  # [B, num_classes]
        return out

class CNN_GRU(nn.Module):
    def __init__(self, input=1,cnn_hidden_size=100, rnn_hidden_size=150, num_layers=3, num_classes=2,out = 10):
        super(CNN_GRU, self).__init__()

        # CNN部分
        self.cnn_size=cnn_hidden_size
        self.conv1 = nn.Conv2d(in_channels=input, out_channels=cnn_hidden_size, kernel_size=(5,5), stride=(1,1))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=cnn_hidden_size, out_channels=cnn_hidden_size, kernel_size=(7,7), stride=(3,3))
        self.pool1 = nn.MaxPool2d(kernel_size=(5,5), stride=(3,3))
        self.se_block=SEBasicBlock(60,60)
        # RNN部分
        self.gru = nn.GRU(cnn_hidden_size, rnn_hidden_size, num_layers, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(rnn_hidden_size, num_classes)
        self.fc1 = nn.Linear(out, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.shape)
        # assert(1==0)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        cnn_output = self.pool1(x)
        cnn_output = cnn_output.view(cnn_output.shape[0],-1)
        # print(cnn_output.shape)
        # print(cnn_output.shape)
        # assert(1==0)
        cnn_output =cnn_output.view(cnn_output.shape[0],self.cnn_size,-1)
        rnn_input = cnn_output.permute(0, 2, 1)
        rnn_out = self.gru(rnn_input)[0]
        # 取LSTM层的最后一个时间步的输出作为模型输出
        output = self.fc(rnn_out[:, -1, :])

        return output
    

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40,input=1):
        # self.patch_size = patch_size
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 9)),
            nn.Conv2d(40, 40, (9, 1), (3, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((3, 7), (3, 7)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.conv=nn.Conv2d(60,24,(1,25),(1,6))
        self.relu=nn.ReLU()


    def forward(self, x) -> Tensor:

        x=x.unsqueeze(-3)

        x = self.shallownet(x)

        x = self.projection(x)

        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self,x, mask = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super(TransformerEncoderBlock,self).__init__()
        self.resadd1 = ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            ))
        self.resadd2 = ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
    def forward(self, x):
        x = self.resadd1(x)
        x = self.resadd2(x)
        return x
        

class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size):
        super(TransformerEncoder,self).__init__()
        self.encoder = []
        for _ in range(depth):
            self.encoder.append(TransformerEncoderBlock(emb_size))
        self.encoder = nn.Sequential(*self.encoder)
    def forward(self, x):
        x = self.encoder(x)
        return x
class PatchEmbedding_spec(nn.Module):
    def __init__(self, emb_size=40,input=1):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 10, (1, 7), (1, 3)),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, (5, 1), (3, 1)),
            nn.BatchNorm2d(10),
            nn.ELU(),
            nn.AvgPool2d((4, 4), (4, 4)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(10, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
    def forward(self, x) -> Tensor:
        # x=x.unsqueeze(-3)
        x = self.shallownet(x)
        x = self.projection(x)
        return x
    
class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super(ClassificationHead,self).__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc1 = nn.Sequential(
            # nn.Linear(840, 256),
            # nn.ELU(),
            # nn.Dropout(0.5),
            nn.Linear(120, 32),
            nn.ELU(),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(300, 32),
            nn.ELU(),
            nn.Dropout(0.3)
        )
        self.fc = nn.Linear(32, n_classes)
        self.fc_spec = nn.Linear(64, n_classes)
    def forward(self, x,de):
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        # 暂时
        # if de!=None:
        #     de = de.contiguous().view(x.size(0), -1)
        #     de = self.fc2(de)
        #     x = torch.cat((x,de),dim=1)
        #     out = self.fc_spec(x)
        # else:
        #     out = self.fc(x)
        out = x
        return out

class Conformer(nn.Module):
    def __init__(self, emb_size=20, depth=2, n_classes=4,input=1, spec_num=0,de=False,**kwargs):
        super(Conformer, self).__init__()
        self.PatchEmbedding=PatchEmbedding(emb_size,input)
        self.PatchEmbedding_spec=PatchEmbedding_spec(int(emb_size),input)
        self.TransformerEncoder1=TransformerEncoder(depth, emb_size)
        self.TransformerEncoder2=TransformerEncoder(depth, int(emb_size))
        self.ClassificationHead=ClassificationHead(emb_size, n_classes)
        self.spec_num = spec_num
        self.de=de

    def forward(self, x,spec=None,de=None):

        x=self.PatchEmbedding(x)
        # print(x.shape)
        # x=self.TransformerEncoder1(x)
        if self.spec_num != 0:
            spec = self.PatchEmbedding_spec(spec)
            # print(spec.shape)
            # assert(1==0)
            x = torch.cat((x,spec),dim=1)
            # spec = self.TransformerEncoder2(spec)
        # spec=self.TransformerEncoder1(spec)
        x=self.TransformerEncoder1(x)
        # x = torch.cat((x,spec),dim=1)
        # x=self.ClassificationHead(spec,None)
        if self.de:
            x=self.ClassificationHead(x,de)
        else:
            x=self.ClassificationHead(x,None)

        return x
    

######################################################################
# EEGNet (compact version)
# Reference: Lawhern et al., 2018
######################################################################
class EEGNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 4, 
                 F1: int = 8, D: int = 2, kern_length: int = 64, dropout_prob: float = 0.5):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F1 * D
        # Temporal conv
        self.temporal = nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length//2), bias=False)
        # Depthwise (spatial) conv across channels
        self.depthwise = nn.Conv2d(F1, self.F2, (in_channels, 1), groups=F1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.F2)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout = nn.Dropout(dropout_prob)
        # Separable conv
        self.separable = nn.Sequential(
            nn.Conv2d(self.F2, self.F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.AvgPool2d((1, 8)),
            nn.Flatten()
        )
        # classifier
        # compute final feature dim adaptively in forward
        self.classifier = None
        self.num_classes = num_classes

    def forward(self, x):
        # x: (B, C, T) -> treat as (B, 1, C, T)
        B, C, T = x.shape
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.depthwise(x)
        x = F.elu(self.bn1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.separable(x)
        if self.classifier is None:
            self.classifier = nn.Linear(x.shape[1], self.num_classes).to(x.device)
        out = self.classifier(x)
        return out


######################################################################
# DeepConvNet (Schirrmeister et al.)
######################################################################
class DeepConvNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            # reshape to (B,1,C,T)
            # block 1
            nn.Conv2d(1, 25, (1, 5), padding=(0,2), bias=False),
            nn.Conv2d(25, 25, (in_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
            # block 2
            nn.Conv2d(25, 50, (1,5), padding=(0,2), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5),
            # block 3
            nn.Conv2d(50, 100, (1,5), padding=(0,2), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5),
            # block 4
            nn.Conv2d(100, 200, (1,5), padding=(0,2), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5),
            nn.Flatten()
        )
        self.classifier = None
        self.num_classes = num_classes

    def forward(self, x):
        B, C, T = x.shape
        x = x.unsqueeze(1)
        x = self.net(x)
        if self.classifier is None:
            self.classifier = nn.Linear(x.shape[1], self.num_classes).to(x.device)
        return self.classifier(x)


######################################################################
# ShallowConvNet
######################################################################
class ShallowConvNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0,12), bias=False),
            nn.Conv2d(40, 40, (in_channels, 1), bias=False),
            nn.BatchNorm2d(40),
            nn.Square(),
            nn.AvgPool2d((1,75)),
            nn.LogSoftmax(dim=1)
        )
        # note: Square is not native; define fallback

class Square(nn.Module):
    def forward(self, x):
        return x**2

# replace usage
ShallowConvNet.net = nn.Sequential(
    nn.Conv2d(1, 40, (1, 25), padding=(0,12), bias=False),
    nn.Conv2d(40, 40, (1, 1), bias=False),
    nn.BatchNorm2d(40),
    Square(),
    nn.AvgPool2d((1,75)),
    nn.Flatten()
)

# Provide classifier in init
def _shallow_init(self, in_channels, num_classes=4):
    nn.Module.__init__(self)
    self.net = ShallowConvNet.net
    self.classifier = None
    self.num_classes = num_classes

ShallowConvNet.__init__ = _shallow_init

def ShallowConvNet_forward(self, x):
    B,C,T = x.shape
    x = x.unsqueeze(1)
    x = self.net(x)
    if self.classifier is None:
        self.classifier = nn.Linear(x.shape[1], self.num_classes).to(x.device)
    return self.classifier(x)

ShallowConvNet.forward = ShallowConvNet_forward


######################################################################
# TSCeption (inception-style temporal convs + spatial fusion)
# A compact architecture using multiple temporal kernel sizes in parallel
######################################################################
class TSCeption(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 4, base_filters: int = 32):
        super().__init__()
        kernel_sizes = [3, 7, 15]
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=k, padding=k//2, groups=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU()
            ))
            self.fuse = nn.Sequential(
            nn.Conv1d(base_filters * len(kernel_sizes), base_filters, kernel_size=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU()
            )
            self.classifier = None
            self.num_classes = num_classes


    def forward(self, x):
        # x: (B, C, T)
        branch_outs = [br(x) for br in self.branches] # each: (B, base_filters, T)
        out = torch.cat(branch_outs, dim=1) # (B, base_filters*len, T)
        out = self.fuse(out) # (B, base_filters, T)
        out = out.mean(dim=2) # (B, base_filters)
        if self.classifier is None:
            self.classifier = nn.Linear(out.shape[1], self.num_classes).to(x.device)
        return self.classifier(out)


######################################################################
# Simple Spatial-Temporal Transformer (STTransformer)
# - Spatial Transformer: self-attention over channels for each time
# - Temporal Transformer: standard Transformer over time on channel-aggregated features
######################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1), :]
        return x

class STTransformer(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 4, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        # project channel dimension to d_model per time step
        self.spatial_proj = nn.Linear(in_channels, d_model)
        # temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True)
        self.temporal_enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.spatial_proj(x)  # (B, T, d_model)
        x = self.pos(x)
        x = self.temporal_enc(x)  # (B, T, d_model)
        # aggregate across time
        x = x.mean(dim=1)
        return self.classifier(x)


######################################################################
# Conformer-like block (simplified)
# Reference: Gulati et al. Conformer: Convolution-augmented Transformer
######################################################################
class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(self, dim, kernel_size=31):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2*dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        # x: (B, T, D)
        x = self.layernorm(x)
        x = x.permute(0,2,1)  # (B, D, T)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise(x)
        x = self.pointwise_conv2(x)
        x = self.bn(x)
        x = x.permute(0,2,1)
        return x

class ConformerBlock(nn.Module):
    def __init__(self, dim, nhead=4, ff_hidden=256, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, ff_hidden, dropout)
        self.mha = nn.MultiheadAttention(dim, num_heads=nhead, batch_first=True, dropout=dropout)
        self.conv = ConformerConvModule(dim, kernel_size=conv_kernel)
        self.ff2 = FeedForwardModule(dim, ff_hidden, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, src_mask=None):
        # x: (B, T, D)
        x = x + 0.5 * self.ff1(x)
        attn_out, _ = self.mha(x, x, x, attn_mask=src_mask)
        x = x + attn_out
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.norm(x)
        return x

class ConformerClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 4, d_model: int = 128, num_blocks: int = 2):
        super().__init__()
        # project channels to d_model features per time step
        self.input_proj = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList([ConformerBlock(d_model) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B,C,T)
        x = x.permute(0,2,1)  # (B, T, C)
        x = self.input_proj(x)  # (B, T, D)
        for b in self.blocks:
            x = b(x)
        x = x.mean(dim=1)
        return self.classifier(x)