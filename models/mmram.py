# -*- coding: utf-8 -*-
# file: mmtan.py
# author: jyu5 <yujianfei1990@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
from layers.mm_attention import MMAttention
import torch
import torch.nn as nn
import torch.nn.functional as F


class MMRAM(nn.Module):
    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                memory[i][idx] *= (1-float(idx)/int(memory_len[i]))
        return memory

    def __init__(self, embedding_matrix, opt):
        super(MMRAM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bi_lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True, dropout = opt.dropout_rate)
        self.bi_lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True, dropout = opt.dropout_rate)
        self.attention = Attention(opt.hidden_dim*2, score_function='mlp', dropout=opt.dropout_rate)
        self.gru_cell = nn.GRUCell(opt.hidden_dim*2, opt.hidden_dim*2)
        if self.opt.tfn:
            self.vis2text = nn.Linear(2048, opt.hidden_dim*2)
            self.dense = nn.Linear(opt.hidden_dim*opt.hidden_dim*4, opt.polarities_dim)
        else:
            self.vis2text = nn.Linear(2048, opt.hidden_dim)
            self.dense = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)

    def forward(self, inputs, visual_embeds_global, visual_embeds_mean, visual_embeds_att, att_mode):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        memory_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)

        memory = self.embed(text_raw_indices)
        memory, (_, _) = self.bi_lstm_context(memory, memory_len)
        # memory = self.locationed_memory(memory, memory_len)
        aspect = self.embed(aspect_indices)
        aspect, (_, _) = self.bi_lstm_aspect(aspect, aspect_len)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))

        et = aspect
        for _ in range(self.opt.hops):
            it_al = self.attention(memory, et, memory_len).squeeze(dim=1)
            et = self.gru_cell(it_al, et)

        converted_vis_embed = self.vis2text(torch.tanh(visual_embeds_global))
        if self.opt.tfn:
            dot_matrix = torch.bmm(et.unsqueeze(2), converted_vis_embed.unsqueeze(1))
            x = dot_matrix.view(-1, self.opt.hidden_dim*self.opt.hidden_dim*4)
            out = self.dense(x)
        else:
            x = torch.cat((et, converted_vis_embed), dim=-1)
            out = self.dense(x)
        return out
        #, _, _, _
