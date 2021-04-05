import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
import copy
from .set_encoder_text import SetEncoder as SetTextEncoder
from .set_encoder import SetEncoder as SetVideoEncoder

"""
    Classes and functions required for Set encoding in adaptation networks. Many of the ideas and classes here are 
    closely related to DeepSets (https://arxiv.org/abs/1703.06114).
"""

def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)

class SetVideoTextEncoder(nn.Module):
    """
    Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
    on sets (mainly for extracting task-level representations from context sets).
    """
    def __init__(self, text_emb, num_layers, ratio, temp_dim, emb):
        super(SetVideoTextEncoder, self).__init__()

        print("Video-Text Encoder")

        self.pre_pooling_fn = SimplePrePoolNet(text_emb, num_layers, ratio, temp_dim, emb)
        self.pooling_fn = mean_pooling

    def forward(self, videos, labels):
        """
        Forward pass through DeepSet SetEncoder. Implements the following computation:

                g(X) = rho ( mean ( phi(x) ) )
                Where X = (x0, ... xN) is a set of elements x in X (in our case, images from a context set)
                and the mean is a pooling operation over elements in the set.

        :param x: (torch.tensor) Set of elements X (e.g., for images has shape batch x C x H x W ).
        :return: (torch.tensor) Representation of the set, single vector in Rk.
        """
        set_text_video_emb = self.pre_pooling_fn(videos, labels)
        prepooling_text_video_emb = set_text_video_emb
        set_text_video_emb = self.pooling_fn(set_text_video_emb)

        return set_text_video_emb, prepooling_text_video_emb

class SimplePrePoolNet(nn.Module):
    """
    Simple prepooling network for images. Implements the phi mapping in DeepSets networks. In this work we use a
    multi-layer convolutional network similar to that in https://openreview.net/pdf?id=rJY0-Kcll.
    """
    def __init__(self, text_emb, num_layers, ratio, temp_dim, emb):
        super(SimplePrePoolNet, self).__init__()
        
        self.set_text_encoder = SetTextEncoder(text_emb, num_layers, ratio, temp_dim, emb)
        self.set_video_encoder = SetVideoEncoder()

        in_dim = self.set_text_encoder.pre_pooling_fn.output_size
        self.norm_layer = nn.LayerNorm(in_dim*2)
        self.layer_emb = nn.Linear(in_dim*2, in_dim)
        self.activation = nn.ReLU()


    def forward(self, videos, labels):

        _, set_emb_text = self.set_text_encoder(labels)
        _, set_emb_video = self.set_video_encoder(videos)

        set_emb = torch.cat((set_emb_text, set_emb_video), 1)
        set_emb = self.norm_layer(set_emb)
        set_emb = self.layer_emb(set_emb)
        set_emb = self.activation(set_emb)

        return set_emb

    @property
    def output_size(self):
        return self.set_text_encoder.pre_pooling_fn.output_size
