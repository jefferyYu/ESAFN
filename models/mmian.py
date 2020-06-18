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


class MMIAN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(MMIAN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                        batch_first=True)  # , dropout = opt.dropout_rate
        self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                       batch_first=True)  # , dropout = opt.dropout_rate
        self.attention_aspect = Attention(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)
        self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)
        # self.attention_aspect = Attention(opt.hidden_dim, score_function='mlp', dropout=opt.dropout_rate)
        # self.attention_context = Attention(opt.hidden_dim, score_function='mlp', dropout=opt.dropout_rate)
        self.vis2text = nn.Linear(2048, opt.hidden_dim)
        self.dense = nn.Linear(opt.hidden_dim * 3, opt.polarities_dim)

    def forward(self, inputs, visual_embeds_global, visual_embeds_mean, visual_embeds_att, att_mode):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        ori_text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        ori_aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        context = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        context_lstm, (_, _) = self.lstm_context(context, ori_text_raw_len)
        aspect_lstm, (_, _) = self.lstm_aspect(aspect, ori_aspect_len)

        aspect_len = torch.tensor(ori_aspect_len, dtype=torch.float).to(self.opt.device)
        sum_aspect = torch.sum(aspect_lstm, dim=1)
        avg_aspect = torch.div(sum_aspect, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = torch.tensor(ori_text_raw_len, dtype=torch.float).to(self.opt.device)
        sum_context = torch.sum(context_lstm, dim=1)
        avg_context = torch.div(sum_context, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final = self.attention_aspect(aspect_lstm, avg_context, ori_aspect_len).squeeze(dim=1)
        context_final = self.attention_context(context_lstm, avg_aspect, ori_text_raw_len).squeeze(dim=1)

        converted_vis_embed = self.vis2text(torch.tanh(visual_embeds_global))

        text_representation = torch.cat((aspect_final, context_final), dim=-1)

        x = torch.cat((text_representation, converted_vis_embed), dim=-1)
        out = self.dense(x)
        return out, _, _, _
