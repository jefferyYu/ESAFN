# -*- coding: utf-8 -*-
# file: mmtan.py
# author: jyu5 <yujianfei1990@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
from layers.mm_attention import MMAttention
from layers.attention2 import Attention2
import torch
import torch.nn as nn
import torch.nn.functional as F


class MMFUSION(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(MMFUSION, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True) #, dropout = opt.dropout_rate
        self.lstm_l = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True) #, dropout = opt.dropout_rate
        self.lstm_r = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)  #, dropout = opt.dropout_rate
        self.attention_l = Attention2(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)
        self.attention_r = Attention2(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)
        self.visaspect_att_l = MMAttention(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)
        self.visaspect_att_r = MMAttention(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)
        self.ltext2hidden = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.laspect2hidden = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.rtext2hidden = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.raspect2hidden = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.dropout = nn.Dropout(self.opt.dropout_rate)
        #self.viscontext_att_aspect = MMAttention(opt.hidden_dim, score_function='mlp', dropout=opt.dropout_rate)
        #self.visaspect_att_context = MMAttention(opt.hidden_dim, score_function='mlp', dropout=opt.dropout_rate)
        
        self.aspect2text = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.vismap2text = nn.Linear(2048, opt.hidden_dim)
        self.vis2text = nn.Linear(2048, opt.hidden_dim)
        self.gate = nn.Linear(2048+4*opt.hidden_dim, opt.hidden_dim)

        self.madality_attetion = nn.Linear(opt.hidden_dim,1)
        
        #blinear interaction between text vectors and image vectors
        #self.text2hidden = nn.Linear(opt.hidden_dim*3, opt.hidden_dim)
        #self.vis2hidden = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        #self.hidden2final = nn.Linear(opt.hidden_dim, opt.hidden_dim)

        #self.text2hiddentext = nn.Linear(opt.hidden_dim*4, opt.hidden_dim*4)
        #self.vis2hiddentext = nn.Linear(opt.hidden_dim, opt.hidden_dim*4)

        self.text2hiddenvis = nn.Linear(opt.hidden_dim * 4, opt.hidden_dim)
        self.vis2hiddenvis = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        
        #self.dense_2 = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)
        #self.dense_3 = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)
        #self.dense_4 = nn.Linear(opt.hidden_dim*4, opt.polarities_dim)
        #self.dense_5 = nn.Linear(opt.hidden_dim*5, opt.polarities_dim)
        #self.dense_10 = nn.Linear(opt.hidden_dim*10, opt.polarities_dim)
        if self.opt.att_mode == 'vis_concat_attimg' or self.opt.att_mode == 'vis_concat':
            self.dense_5 = nn.Linear(opt.hidden_dim*5, opt.polarities_dim)
        elif self.opt.att_mode == 'vis_concat_attimg_gate':
            if self.opt.tfn:
                self.dense_6 = nn.Linear(opt.hidden_dim * opt.hidden_dim, opt.polarities_dim)
            else:
                self.dense_6 = nn.Linear(opt.hidden_dim*6, opt.polarities_dim)
        #self.dense_7 = nn.Linear(opt.hidden_dim*7, opt.polarities_dim)
        #self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        
    def attention_linear(self, text, converted_vis_embed_map, vis_embed_map):
        #text: batch_size, hidden_dim; converted_vis_embed_map: batch_size, keys_number,embed_size; vis_embed_map: batch_size, keys_number, 2048
        keys_size = converted_vis_embed_map.size(1)
        text = text.unsqueeze(1).expand(-1, keys_size, -1)#batch_size, keys_number,hidden_dim
        attention_inputs = torch.tanh(text + converted_vis_embed_map)
        #attention_inputs = F.dropout( attention_inputs )
        att_weights = self.madality_attetion(attention_inputs).squeeze(2) #batch_size, keys_number
        att_weights = F.softmax(att_weights, dim=-1).view(-1,1,49) #batch_size, 1, keys_number

        att_vector = torch.bmm(att_weights, vis_embed_map).view(-1, 2048) #batch_size, 2048
        return att_vector, att_weights

    def forward(self, inputs, visual_embeds_global, visual_embeds_mean, visual_embeds_att, att_mode):
        x_l, x_r, aspect_indices = inputs[0], inputs[1], inputs[2]
        ori_x_l_len = torch.sum(x_l != 0, dim=-1)
        ori_x_r_len = torch.sum(x_r != 0, dim=-1)
        ori_aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        aspect = self.embed(aspect_indices)
        aspect_lstm, (_, _) = self.lstm_aspect(aspect, ori_aspect_len)
        aspect_len = torch.tensor(ori_aspect_len, dtype=torch.float).to(self.opt.device)
        sum_aspect = torch.sum(aspect_lstm, dim=1)
        avg_aspect = torch.div(sum_aspect, aspect_len.view(aspect_len.size(0), 1))
        
        # obtain the lstm hidden states for the left context and the right context respectively
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        l_context, (_, _) = self.lstm_l(x_l, ori_x_l_len)
        r_context, (_, _) = self.lstm_r(x_r, ori_x_r_len)
        
        converted_vis_embed = self.vis2text(torch.tanh(visual_embeds_global))
        
        if att_mode == 'text': # apply aspect words to attend the left and right contexts
            l_mid, l_att = self.attention_l(l_context, avg_aspect, ori_x_l_len)
            r_mid, r_att = self.attention_r(r_context, avg_aspect, ori_x_r_len)
            l_final = l_mid.squeeze(dim=1)
            r_final = r_mid.squeeze(dim=1)
            
            context = torch.cat((l_final, r_final), dim=-1)
            x = torch.cat((context, avg_aspect), dim=-1)
            out = self.dense_3(x)
            return out
        elif att_mode == 'vis_only': #only use image and aspect words
            vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)#self.batch_size, 49, 2048
            converted_vis_embed_map = self.vismap2text(vis_embed_map) #self.batch_size, 49, embed
            converted_aspect = self.aspect2text(avg_aspect) 
        
            #att_vector: batch_size, 2048
            att_vector, att_weights = self.attention_linear(converted_aspect, converted_vis_embed_map, vis_embed_map)
            converted_att_vis_embed = self.vis2text(torch.tanh(att_vector))
            x = torch.cat((avg_aspect, converted_att_vis_embed), dim=-1)
            #x = torch.cat((avg_aspect, converted_vis_embed), dim=-1)
            out = self.dense_2(x)
            return out
        elif att_mode == 'vis_concat': # "text" mode concatenated with image
            l_mid, l_att = self.attention_l(l_context, avg_aspect, ori_x_l_len)
            r_mid, r_att = self.attention_r(r_context, avg_aspect, ori_x_r_len)
            l_final = l_mid.squeeze(dim=1)
            r_final = r_mid.squeeze(dim=1)

            #"""
            # low-rank pooling
            l_text = torch.tanh(self.ltext2hidden(l_final))  # batch_size, hidde_dim
            l_aspect = torch.tanh(self.laspect2hidden(avg_aspect))  # batch_size, hidden_dim
            ltext_aspect_inter = torch.mul(l_text, l_aspect)
            l_output = torch.cat((ltext_aspect_inter, l_final), dim=-1)
            #l_output = ltext_aspect_inter + l_final

            r_text = torch.tanh(self.rtext2hidden(r_final))  # batch_size, hidde_dim
            r_aspect = torch.tanh(self.raspect2hidden(avg_aspect))  # batch_size, hidden_dim
            rtext_aspect_inter = torch.mul(r_text, r_aspect)
            r_output = torch.cat((rtext_aspect_inter, r_final), dim=-1)
            #r_output = rtext_aspect_inter + r_final

            text_representation = torch.cat((l_output, r_output), dim=-1)

            x = torch.cat((text_representation, converted_vis_embed), dim=-1)
            
            x = self.dropout(x)

            out = self.dense_5(x)
            return out
        elif att_mode == 'vis_concat_attimg':  # "text" mode concatenated with attention-based image
            l_mid, l_att = self.attention_l(l_context, avg_aspect, ori_x_l_len)
            r_mid, r_att = self.attention_r(r_context, avg_aspect, ori_x_r_len)
            l_final = l_mid.squeeze(dim=1)
            r_final = r_mid.squeeze(dim=1)

            #"""
            # low-rank pooling
            l_text = torch.tanh(self.ltext2hidden(l_final))  # batch_size, hidde_dim
            l_aspect = torch.tanh(self.laspect2hidden(avg_aspect))  # batch_size, hidden_dim
            ltext_aspect_inter = torch.mul(l_text, l_aspect)
            l_output = torch.cat((ltext_aspect_inter, l_final), dim=-1)
            #l_output = ltext_aspect_inter + l_final

            r_text = torch.tanh(self.rtext2hidden(r_final))  # batch_size, hidde_dim
            r_aspect = torch.tanh(self.raspect2hidden(avg_aspect))  # batch_size, hidden_dim
            rtext_aspect_inter = torch.mul(r_text, r_aspect)
            r_output = torch.cat((rtext_aspect_inter, r_final), dim=-1)
            #r_output = rtext_aspect_inter + r_final

            text_representation = torch.cat((l_output, r_output), dim=-1)

            vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048
            converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, embed
            converted_aspect = self.aspect2text(avg_aspect)

            # att_vector: batch_size, 2048
            att_vector, att_weights = self.attention_linear(converted_aspect, converted_vis_embed_map, vis_embed_map)
            converted_att_vis_embed = self.vis2text(torch.tanh(att_vector))

            x = torch.cat((text_representation, converted_att_vis_embed), dim=-1)
            x = self.dropout(x)

            out = self.dense_5(x)
            return out
        elif att_mode == 'vis_concat_attimg_gate':  # "text" mode concatenated with gated attention-based image
            l_mid, l_att = self.attention_l(l_context, avg_aspect, ori_x_l_len)
            r_mid, r_att = self.attention_r(r_context, avg_aspect, ori_x_r_len)
            l_final = l_mid.squeeze(dim=1)
            r_final = r_mid.squeeze(dim=1)

            #context = torch.cat((l_final, r_final), dim=-1)
            #text_representation = torch.cat((context, avg_aspect), dim=-1)

            #"""
            # low-rank pooling
            l_text = torch.tanh(self.ltext2hidden(l_final))  # batch_size, hidde_dim
            l_aspect = torch.tanh(self.laspect2hidden(avg_aspect))  # batch_size, hidden_dim
            ltext_aspect_inter = torch.mul(l_text, l_aspect)
            l_output = torch.cat((ltext_aspect_inter, l_final), dim=-1)
            #l_output = ltext_aspect_inter + l_final

            r_text = torch.tanh(self.rtext2hidden(r_final))  # batch_size, hidde_dim
            r_aspect = torch.tanh(self.raspect2hidden(avg_aspect))  # batch_size, hidden_dim
            rtext_aspect_inter = torch.mul(r_text, r_aspect)
            r_output = torch.cat((rtext_aspect_inter, r_final), dim=-1)
            #r_output = rtext_aspect_inter + r_final

            text_representation = torch.cat((l_output, r_output), dim=-1)
            #text_representation = torch.cat((text_representation, avg_aspect), dim=-1)
            #text_representation = self.dropout(text_representation)
            #context = torch.cat((l_output, r_output), dim=-1)
            #text_representation = torch.cat((context, avg_aspect), dim=-1)
            #"""

            # apply entity-based attention mechanism to obtain different image representations
            vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048
            converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, embed
            converted_aspect = self.aspect2text(avg_aspect)

            # att_vector: batch_size, 2048
            att_vector, att_weights = self.attention_linear(converted_aspect, converted_vis_embed_map, vis_embed_map)
            converted_att_vis_embed = self.vis2text(torch.tanh(att_vector))  # att_vector: batch_size, hidden_dim

            merge_representation = torch.cat((text_representation, att_vector), dim=-1)
            gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, hidden_dim
            gated_converted_att_vis_embed = torch.mul(gate_value, converted_att_vis_embed)
            #gated_converted_att_vis_embed = self.dropout(gated_converted_att_vis_embed)

            #"""
            # low-rank pooling
            #text_text = torch.tanh(self.text2hiddentext(text_representation))  # batch_size, hidde_dim
            #vis_text = torch.tanh(self.vis2hiddentext(gated_converted_att_vis_embed))  # batch_size, hidden_dim
            #vis_text_inter = torch.mul(text_text, vis_text)
            #text_output = torch.cat((text_representation, vis_text_inter), dim =-1)
            #text_output = text_representation + vis_text_inter
            #"""

            #"""
            text_vis = torch.tanh(self.text2hiddenvis(text_representation))  # batch_size, hidde_dim
            vis_vis = torch.tanh(self.vis2hiddenvis(gated_converted_att_vis_embed))  # batch_size, hidden_dim

            if self.opt.tfn:
                dot_matrix = torch.bmm(text_vis.unsqueeze(2), vis_vis.unsqueeze(1))
                x = dot_matrix.view(-1, self.opt.hidden_dim * self.opt.hidden_dim)
                x = self.dropout(x)
                out = self.dense_6(x)
            else:
                text_vis_inter = torch.mul(text_vis, vis_vis)
                vis_output = torch.cat((gated_converted_att_vis_embed, text_vis_inter), dim=-1)

                # comb = torch.cat((text_representation, gated_converted_att_vis_embed), dim=-1)
                # x = torch.cat((comb, text_vis_inter), dim=-1)
                x = torch.cat((text_representation, vis_output), dim=-1)
                x = self.dropout(x)
                out = self.dense_6(x)
            #"""

            """
            without Multimodal Fusion (MF)
            vis_output = gated_converted_att_vis_embed
            #vis_output = gated_converted_att_vis_embed + text_vis_inter
            x = torch.cat((text_representation, vis_output), dim=-1)
            x = self.dropout(x)
            out = self.dense_5(x)
            """

            return out, l_att, r_att, att_weights

        elif att_mode == 'vis_att': # apply aspect words and image to attend the left and right contexts
            l_mid,_ = self.visaspect_att_l(l_context, avg_aspect, converted_vis_embed, \
                                                        ori_x_l_len)
            r_mid,_ = self.visaspect_att_r(r_context, avg_aspect, converted_vis_embed, \
                                                        ori_x_r_len)
            l_final = l_mid.squeeze(dim=1)
            r_final = r_mid.squeeze(dim=1)
            
            context = torch.cat((l_final, r_final), dim=-1)
            x = torch.cat((context, avg_aspect), dim=-1)
            
            out = self.dense_3(x)
            return out