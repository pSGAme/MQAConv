import copy
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.modules import Module
from torch.nn.modules.container import ModuleList
from torch import einsum
import torch
import torch.nn as nn

class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of feature matching and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).

    Examples::
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, dim_feedforward=2048)
        >>> memory = torch.rand(10, 24, 8, 512)
        >>> tgt = torch.rand(20, 24, 8, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, seq_len, d_model=512, dim_feedforward=2048):
        super(TransformerDecoderLayer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.fc1 = nn.Linear(d_model, d_model)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(self.seq_len, dim_feedforward)
        self.bn2 = nn.BatchNorm1d(dim_feedforward)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(dim_feedforward, 1)
        self.bn3 = nn.BatchNorm1d(1)
        self.fc = nn.Linear(self.seq_len, 1)
        score_embed = torch.randn(seq_len, seq_len)
        score_embed = score_embed + score_embed.t()
        self.score_embed = nn.Parameter(score_embed.view(1,1,seq_len,seq_len))


    def forward(self, tgt: Tensor, memory: Tensor, flag) -> Tensor:
        r"""Pass the inputs through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).

        Shape:
            tgt: [q, h, w, d], where q is the query length, d is d_model, and (h, w) is feature map size
            memory: [k, h, w, d], where k is the memory length
        """
        q, h, w, d = tgt.size()
        k, h, w, d = memory.size()

        tgt = tgt.view(q, -1, d)  # b hw d
        memory = memory.view(k, -1, d) # b hw
        #print(flag)
        if flag:
            query = self.fc1(tgt)
            key = self.fc1(memory)
            score = einsum('q t d, k s d -> q k s t', query, key) * self.score_embed.sigmoid()  # b b hw hw
        else:
            query = tgt
            key = memory
            score = einsum('q t d, k s d -> q k s t', query, key)  # b b hw hw
        score = score.reshape(q * k, self.seq_len, self.seq_len) # bb hw hw
        score = torch.cat((score.max(dim=1)[0], score.max(dim=2)[0]), dim=-1) # b b 2hw
        score = score.view(-1, 1, self.seq_len) # b*b*2, hw
        score = self.bn1(score).view(-1, self.seq_len) # bb 2, hw


        score = self.fc2(score)
        score = self.bn2(score)
        score = self.relu(score)
        score = self.fc3(score) # bb2 ,1
        # score = self.fc(score)

        score = score.view(-1, 2).sum(dim=-1, keepdim=True) # bb 2 sum to bb 1
        score = self.bn3(score)
        score = score.view(q, k)
        return score


class TransformerDecoder(Module):

    __constants__ = ['norm']

    def __init__(self, decoder_layer, split_layers, norm=None, use_transformer=False):
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(len(split_layers))])
        self.split_layers = split_layers
        self.norm = norm
        self.use_transformer = use_transformer

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        """
            Shape:
            tgt: [q, h, w, d*n], where q is the query length, d is d_model, n is num_layers, and (h, w) is feature map size
            memory: [k, h, w, d*n], where k is the memory length
        """
        #  memory, features
        # b, h, w, 3*c  self.num_layers = 3

        tgt = tgt.split(self.split_layers, dim=-1) #
        memory = memory.split(self.split_layers, dim=-1)
        score = None
        #print(len(memory))
        for i, mod in enumerate(self.layers):
            if i == 0:
                #score = mod(tgt[i], memory[i], self.use_transformer and (i==num_layer-1 or i==num_layer-2))
                score = mod(tgt[i], memory[i],False)
            else:
                score += mod(tgt[i], memory[i], False)
                #score = score + mod(tgt[i], memory[i], self.use_transformer and (i==num_layer-1 or i==num_layer-2))
        #print(self.num_layers)
        if self.norm is not None and len(self.split_layers)>1:
            q, k = score.size()
            score = score.view(-1, 1)
            score = self.norm(score)
            score = score.view(q, k)
        return score


class TransMatcher(nn.Module):

    def __init__(self, seq_len, d_model=512, split_layer=None, dim_feedforward=2048, use_transformer = False):
        super().__init__()
        if split_layer is None:
            split_layer = [512]
        self.seq_len = seq_len
        self.d_model = d_model

        self.decoder_layer = TransformerDecoderLayer(seq_len, d_model, dim_feedforward)
        decoder_norm = nn.BatchNorm1d(1)
        self.decoder = TransformerDecoder(self.decoder_layer, split_layer, decoder_norm, use_transformer)
        self.memory = None

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def make_kernel(self, features):
        self.memory = features

    def forward(self, features):
        score = self.decoder(self.memory, features)
        return score


