# -*- coding: utf-8 -*- used for returning attention weights
# file: attention.py
# author: jianfei yu <jyu5@snapchat.com>
# Copyright (C) 2018. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attention2(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, n_head=1, score_function='scaled_dot_product', dropout=0.1):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        '''
        super(Attention2, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_function = score_function
        self.w_kx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.w_qx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.proj = nn.Linear(n_head * hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2, 1))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, k, q, memory_len):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head, ?*k_len, embed_dim) -> (n_head*?, k_len, hidden_dim)
        # qx: (n_head, ?*q_len, embed_dim) -> (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, embed_dim,)
        kx = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, ?*k_len, embed_dim)
        qx = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, ?*q_len, embed_dim)
        kx = torch.bmm(kx, self.w_kx).view(-1, k_len, self.hidden_dim)  # (n_head*?, k_len, hidden_dim)
        qx = torch.bmm(qx, self.w_qx).view(-1, q_len, self.hidden_dim)  # (n_head*?, q_len, hidden_dim)
        if self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = F.tanh(torch.matmul(kq, self.weight).squeeze(dim=-1))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = F.tanh(torch.bmm(qw, kt))
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        attentions = torch.squeeze(score, dim=1)
        #print(attentions[:2])
        # create mask based on the sentence lengths
        mask = Variable(torch.ones(attentions.size())).to(self.device)
        for i, l in enumerate(memory_len): 
            if l < k_len:
                mask[i, l:] = 0
        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        #print(masked[:2])
        #print(masked.shape)
        _sums = masked.sum(-1)  # sums per row
        attentions = torch.div(masked, _sums.view(_sums.size(0), 1))
        #print(attentions[:2])
        
        score = torch.unsqueeze(attentions, dim=1)

        output = torch.bmm(score, kx)  # (n_head*?, k_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, k_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, k_len, embed_dim)
        output = self.dropout(output)
        return output,attentions


class SelfAttention(Attention2):
    '''q is a parameter'''
    def __init__(self, embed_dim, hidden_dim=None, n_head=1, score_function='scaled_dot_product', q_len=1, dropout=0.1):
        super(SelfAttention, self).__init__(embed_dim, hidden_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.FloatTensor(q_len, embed_dim))

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(SelfAttention, self).forward(k, q)
