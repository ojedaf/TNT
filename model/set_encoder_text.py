import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
import copy

"""
    Classes and functions required for Set encoding in adaptation networks. Many of the ideas and classes here are 
    closely related to DeepSets (https://arxiv.org/abs/1703.06114).
"""

def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)


class SetEncoder(nn.Module):
    """
    Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
    on sets (mainly for extracting task-level representations from context sets).
    """
    def __init__(self, text_emb, num_layers, ratio, temp_dim, emb, output_dim_model, text_encoder):
        super(SetEncoder, self).__init__()

        print("Text emb: ",text_emb)
        print("Num layers: ",num_layers)
        print("output dim model: ",output_dim_model)
        print("text_encoder: ",text_encoder)

        self.text_emb = text_emb
        self.text_encoder = text_encoder
        self.pre_pooling_fn = SimplePrePoolNet(text_emb, num_layers, output_dim_model, text_encoder)
        if self.text_emb == 'word_level' and text_encoder != 'glove':
            print("Ratio: ",ratio)
            print("Temp dim: ",temp_dim)
            print("Embedding size: ",emb)
            self.att_module = AttModule(ratio, temp_dim, emb)

        self.pooling_fn = mean_pooling

    def forward(self, x):
        """
        Forward pass through DeepSet SetEncoder. Implements the following computation:

                g(X) = rho ( mean ( phi(x) ) )
                Where X = (x0, ... xN) is a set of elements x in X (in our case, images from a context set)
                and the mean is a pooling operation over elements in the set.

        :param x: (torch.tensor) Set of elements X (e.g., for images has shape batch x C x H x W ).
        :return: (torch.tensor) Representation of the set, single vector in Rk.
        """
        x, x_old = self.pre_pooling_fn(x)
        if self.text_emb == 'word_level' and self.text_encoder != 'glove':
            att = self.att_module(x)
            att = att.expand_as(x)
            x = x*att
            x = torch.sum(x, dim = 1)
        # prepooling_emb = x
        x = self.pooling_fn(x)

        return x, x_old

class ContextModule(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(ContextModule, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(ContextModule, self).__setstate__(state)

    def forward(self, src):
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AttModule(nn.Module):
    def __init__(self, ratio, temp_dim, emb):
        super(AttModule, self).__init__()
        out_temp_dim = int(temp_dim/ratio)

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(temp_dim, out_temp_dim),
            nn.ReLU(),
            nn.Linear(out_temp_dim, temp_dim)
        )

        self.maxPool = nn.MaxPool1d(emb, stride=emb)
        self.avgPool = nn.AvgPool1d(emb, stride=emb)
    
    def forward(self, x):

        x_max = self.maxPool(x)
        x_max = self.mlp(x_max)

        x_avg = self.avgPool(x)
        x_avg = self.mlp(x_avg)

        x_sum = x_max + x_avg

        scale = F.sigmoid(x_sum)

        return scale.unsqueeze(2)


class SimplePrePoolNet(nn.Module):
    """
    Simple prepooling network for images. Implements the phi mapping in DeepSets networks. In this work we use a
    multi-layer convolutional network similar to that in https://openreview.net/pdf?id=rJY0-Kcll.
    """
    def __init__(self, text_emb, num_layers, output_dim_model, text_encoder):
        super(SimplePrePoolNet, self).__init__()
        self.num_layers = num_layers
        self.text_emb = text_emb
        self.text_encoder = text_encoder
        if self.text_emb == 'word_level':
            if text_encoder == 'glove':
                self.layer1 = self._make_linear_layer(300, output_dim_model)  # Roberta base
            else:
                self.model = RobertaModel.from_pretrained('roberta-base')
                self.layer1 = self._make_linear_layer(768, output_dim_model)  # Roberta base
        else:
            if text_encoder == 'roberta_large':
                self.layer1 = self._make_linear_layer(1024, output_dim_model)  # Roberta large
            else:
                self.layer1 = self._make_linear_layer(768, output_dim_model)  # Roberta base
        # self.layers = self._get_clones(self._make_linear_layer(64, 64), num_layers)

        self.layer2 = self._make_linear_layer(output_dim_model, 64)
        # self.layer3 = self._make_linear_layer(64, 64)

        # for param in self.model.parameters():
        #     param.requires_grad = False
    
    @staticmethod
    def _get_clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    @staticmethod
    def _make_linear_layer(in_maps, out_maps):
        return nn.Sequential(
            nn.Linear(in_maps, out_maps),
            nn.ReLU()
        )

    def forward(self, x):
        if self.text_emb == 'word_level' and self.text_encoder != 'glove':
            x = self.model(x)[0]
        x = self.layer1(x)
        x_old = x
        # for i in range(self.num_layers):
        #     x = self.layers[i](x)

        x = self.layer2(x)
        # x = self.layer2(x)
        return x, x_old

    @property
    def output_size(self):
        return 64
