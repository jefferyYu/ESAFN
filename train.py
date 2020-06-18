# -*- coding: utf-8 -*-
# file: train.py
# author: jyu5 <yujianfei1990@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from data_utils import ABSADatesetReader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
import argparse
import resnet.resnet as resnet
from resnet.resnet_utils import myResnet
import os
import json
import random

from torchvision import transforms
from models.mmian import MMIAN
from models.mmram import MMRAM
from models.mmmgan import MMMGAN
from models.mmfusion import MMFUSION
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro')
    #f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro
    
def save_checkpoint(state, track_list, filename):
    """
    save checkpoint
    """
    with open(filename+'.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename+'.model')

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print('> training arguments:')
        for arg in vars(opt):
            print('>>> {0}: {1}'.format(arg, getattr(opt, arg)))
            
        if not os.path.exists(opt.checkpoint):
            os.mkdir(opt.checkpoint)
            
        # torch.manual_seed(opt.rand_seed)
        # torch.cuda.manual_seed_all(opt.rand_seed)

        transform = transforms.Compose([
            transforms.RandomCrop(opt.crop_size), #args.crop_size, by default it is set to be 224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
                            
        absa_dataset = ABSADatesetReader(transform, dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len, \
                        path_image=opt.path_image)
        self.train_data_loader = DataLoader(dataset=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.dev_data_loader = DataLoader(dataset=absa_dataset.dev_data, batch_size=opt.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(dataset=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False)
        #self.writer = SummaryWriter(log_dir=opt.logdir)

        print('building model')
        net = getattr(resnet, 'resnet152')()
        net.load_state_dict(torch.load(os.path.join(opt.resnet_root,'resnet152.pth')))
        self.encoder = myResnet(net, opt.fine_tune_cnn, self.opt.device).to(device)
        self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
                if len(p.shape) > 1:
                    self.opt.initializer(p)
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        text_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if self.opt.fine_tune_cnn:
            params = list(text_params) + list(self.encoder.parameters())
        else:
            print('parameters only include text parts without word embeddings')
            params = list(text_params)
        
        optimizer = self.opt.optimizer(params, lr=self.opt.learning_rate)
        
        if self.opt.load_check_point:
            if os.path.isfile(self.opt.checkpoint + self.opt.dataset+ '_'+ self.opt.model_name+ '_'+self.opt.att_mode+'.model'):
                print("loading checkpoint: '{}'".format(self.opt.checkpoint +self.opt.dataset+ '_'+ self.opt.model_name+ '_'+ self.opt.att_mode+'.model'))
                checkpoint_file = torch.load(self.opt.checkpoint +self.opt.dataset+ '_'+ self.opt.model_name+ '_'+ self.opt.att_mode+'.model')
        
                self.model.load_state_dict(checkpoint_file['state_dict'])
                self.encoder.load_state_dict(checkpoint_file['encode_dict'])
                #optimizer.load_state_dict(checkpoint_file['optimizer'])
                if self.opt.load_opt:
                    optimizer.load_state_dict(checkpoint_file['optimizer'])
                    
        if self.opt.load_check_point:
            self.encoder.eval()
            self.model.eval()
            
            n_train_correct, n_train_total = 0, 0
            n_dev_correct, n_dev_total = 0, 0
            n_test_correct, n_test_total = 0, 0
            with torch.no_grad():
                # true_label = None
                # pred_outputs = None
                true_label_list = []
                pred_label_list = []
                for t_batch, t_sample_batched in enumerate(self.dev_data_loader):
                    t_inputs = [t_sample_batched[col].to(device) for col in self.opt.inputs_cols]
                    t_targets = t_sample_batched['polarity'].to(device)
                    t_images = t_sample_batched['image'].to(device)
                    
                    t_imgs_f, t_img_mean, t_img_att = self.encoder(t_images)
                    
                    if self.opt.att_mode!='vis_att_img_gate' and self.opt.att_mode!= 'vis_concat_attimg_gate':
                        t_outputs = self.model(t_inputs, t_imgs_f, t_img_mean, t_img_att, self.opt.att_mode)
                    else:
                        t_outputs,_,_,_ = self.model(t_inputs, t_imgs_f, t_img_mean, t_img_att, self.opt.att_mode)
                        
                    n_dev_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                    n_dev_total += len(t_outputs)

                    true_label_list.append(t_targets.detach().cpu())
                    pred_label_list.append(t_outputs.detach().cpu())

                    # if true_label is None:
                    #     true_label = t_targets.detach().cpu()
                    #     pred_outputs = t_outputs.detach().cpu()
                    # else:
                    #     true_label = np.concatenate((true_label, t_targets.detach().cpu()))
                    #     pred_outputs = np.concatenate((pred_outputs, t_outputs.detach().cpu()))

                true_label = np.concatenate(true_label_list)
                pred_outputs = np.concatenate(pred_label_list)
                
                print('total dev files: '+str(n_dev_total))
                dev_p, dev_r, dev_f1 = macro_f1(true_label, pred_outputs)
                dev_acc = n_dev_correct / n_dev_total             
                
                # true_label = None
                # pred_outputs = None
                true_label_list = []
                pred_label_list = []
                if self.opt.att_mode == 'vis_att_img_gate' or self.opt.att_mode == 'vis_concat_attimg_gate':
                    if not os.path.exists(self.opt.att_file):
                        os.mkdir(self.opt.att_file)
                    fout_l = open(self.opt.att_file+os.sep+self.opt.dataset+ '_'+ self.opt.model_name+'_'+self.opt.att_mode+ '_'+'left_att.txt', 'w')
                    fout_r = open(self.opt.att_file+os.sep+self.opt.dataset+ '_'+ self.opt.model_name+'_'+self.opt.att_mode+ '_'+'right_att.txt', 'w')
                    fout = open(self.opt.att_file+os.sep+self.opt.dataset+ '_'+ self.opt.model_name+'_'+self.opt.att_mode+'_'+'vis_att.txt', 'w')
                                    
                if not os.path.exists(self.opt.pred_file):
                    os.mkdir(self.opt.pred_file)
                fout_p = open(self.opt.pred_file+os.sep+self.opt.dataset+ '_'+ self.opt.model_name+ '_'+ self.opt.att_mode+ '_'+'pred.txt', 'w')
                fout_t = open(self.opt.pred_file+os.sep+self.opt.dataset+ '_'+ self.opt.model_name+ '_'+ self.opt.att_mode+ '_'+'true.txt', 'w')
                
                for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                    t_inputs = [t_sample_batched[col].to(device) for col in self.opt.inputs_cols]
                    t_targets = t_sample_batched['polarity'].to(device)
                    t_images = t_sample_batched['image'].to(device)
                    
                    t_imgs_f, t_img_mean, t_img_att = self.encoder(t_images)
                    
                    if self.opt.att_mode!='vis_att_img_gate' and self.opt.att_mode!='vis_concat_attimg_gate':
                        t_outputs = self.model(t_inputs, t_imgs_f, t_img_mean, t_img_att, self.opt.att_mode)
                        t_preds = torch.argmax(t_outputs, -1)
                        np_preds = np.asarray(t_preds.detach().cpu())
                        np_targets = np.asarray(t_targets.detach().cpu())
                        for i in range(len(np_preds)):
                            attstr = str(np_preds[i])
                            fout_p.write(attstr+'\n')
                        for i in range(len(np_targets)):
                            attstr = str(np_targets[i])
                            fout_t.write(attstr+'\n')
                    else:
                        t_outputs,l_att,r_att,att_weights = self.model(t_inputs, t_imgs_f, t_img_mean, t_img_att, self.opt.att_mode)
                        t_preds = torch.argmax(t_outputs, -1)
                        l_att = np.asarray(l_att.detach().cpu())
                        r_att = np.asarray(r_att.detach().cpu())
                        np_preds = np.asarray(t_preds.detach().cpu())
                        np_targets = np.asarray(t_targets.detach().cpu())
                        att_weights = np.asarray(att_weights.detach().cpu())
                        for i in range(len(l_att)):
                            attstr = ' '.join(map(str, l_att[i]))
                            fout_l.write(attstr+'\n')
                        for i in range(len(r_att)):
                            attstr = ' '.join(map(str, r_att[i]))
                            fout_r.write(attstr+'\n')
                        for i in range(len(att_weights)):
                            attstr = ' '.join(map(str, att_weights[i]))
                            fout.write(attstr+'\n')
                        for i in range(len(np_preds)):
                            attstr = str(np_preds[i])
                            fout_p.write(attstr+'\n')
                        for i in range(len(np_targets)):
                            attstr = str(np_targets[i])
                            fout_t.write(attstr+'\n')
                        
                    n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                    n_test_total += len(t_outputs)

                    true_label_list.append(t_targets.detach().cpu())
                    pred_label_list.append(t_outputs.detach().cpu())

                    # if true_label is None:
                    #     true_label = t_targets.detach().cpu()
                    #     pred_outputs = t_outputs.detach().cpu()
                    # else:
                    #     true_label = np.concatenate((true_label, t_targets.detach().cpu()))
                    #     pred_outputs = np.concatenate((pred_outputs, t_outputs.detach().cpu()))

                true_label = np.concatenate(true_label_list)
                pred_outputs = np.concatenate(pred_label_list)
                
                if self.opt.att_mode == 'vis_att_img_gate' or self.opt.att_mode == 'vis_concat_attimg_gate':
                    fout_l.close()
                    fout_r.close()
                    fout.close()
                fout_p.close()
                fout_t.close()
                
                print('total test files: '+str(n_test_total))
                test_p, test_r, test_f1  = macro_f1(true_label, pred_outputs)
                test_acc = n_test_correct / n_test_total

                print('dev_acc: {:.4f}, test_acc: {:.4f}'.format(dev_acc, test_acc))
                print('dev_p: {:.4f}, dev_r: {:.4f}, dev_f1: {:.4f}'.format(dev_p, dev_r, dev_f1))
                print('test_p: {:.4f}, test_r: {:.4f}, test_f1: {:.4f}'.format(test_p, test_r, test_f1))
        else:
            max_dev_acc = 0
            max_dev_p = 0
            max_dev_r = 0
            max_dev_f1 = 0
            max_test_p = 0
            max_test_r = 0
            max_test_f1 = 0
            max_test_acc = 0
            global_step = 0
            track_list = list()
            
            for epoch in range(self.opt.num_epoch):
                print('>' * 100)
                print('epoch: ', epoch)
                n_correct, n_total = 0, 0
                for i_batch, sample_batched in enumerate(self.train_data_loader):
                    global_step += 1

                    # switch model to training mode, clear gradient accumulators
                    self.encoder.train()
                    self.model.train()
                    optimizer.zero_grad()
                    self.encoder.zero_grad()

                    inputs = [sample_batched[col].to(device) for col in self.opt.inputs_cols]
                    targets = sample_batched['polarity'].to(device)
                    images = sample_batched['image'].to(device)
                    with torch.no_grad():
                        imgs_f, img_mean, img_att = self.encoder(images)
                    
                    if self.opt.att_mode!='vis_att_img_gate' and self.opt.att_mode!='vis_concat_attimg_gate':
                        outputs = self.model(inputs, imgs_f, img_mean, img_att, self.opt.att_mode)
                    else:
                        outputs,_,_,_ = self.model(inputs, imgs_f, img_mean, img_att, self.opt.att_mode)

                    loss = criterion(outputs, targets)
                    loss.backward()
                
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    if self.opt.att_mode != 'text':
                        nn.utils.clip_grad_norm(params, self.opt.clip_grad)
                    optimizer.step()

                    if global_step % self.opt.log_step == 0:
                    #if i_batch == len(self.train_data_loader)-1:
                        # switch model to evaluation mode
                        self.encoder.eval()
                        self.model.eval()
                    
                        n_train_correct, n_train_total = 0, 0
                        n_dev_correct, n_dev_total = 0, 0
                        n_test_correct, n_test_total = 0, 0
                        with torch.no_grad():
                            for t_batch, t_sample_batched in enumerate(self.train_data_loader):
                                t_inputs = [t_sample_batched[col].to(device) for col in self.opt.inputs_cols]
                                t_targets = t_sample_batched['polarity'].to(device)
                                t_images = t_sample_batched['image'].to(device)
                            
                                t_imgs_f, t_img_mean, t_img_att = self.encoder(t_images)
                            
                                if self.opt.att_mode!='vis_att_img_gate' and self.opt.att_mode!='vis_concat_attimg_gate':
                                    t_outputs = self.model(t_inputs, t_imgs_f, t_img_mean, t_img_att, self.opt.att_mode)
                                else:
                                    t_outputs,_,_,_ = self.model(t_inputs, t_imgs_f, t_img_mean, t_img_att, self.opt.att_mode)

                                n_train_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                                n_train_total += len(t_outputs)
                            train_acc = n_train_correct/n_train_total
                            
                            # true_label = None
                            # pred_outputs = None
                            true_label_list = []
                            pred_label_list = []
                            for t_batch, t_sample_batched in enumerate(self.dev_data_loader):
                                t_inputs = [t_sample_batched[col].to(device) for col in self.opt.inputs_cols]
                                t_targets = t_sample_batched['polarity'].to(device)
                                t_images = t_sample_batched['image'].to(device)
                            
                                t_imgs_f, t_img_mean, t_img_att = self.encoder(t_images)
                            
                                if self.opt.att_mode!='vis_att_img_gate' and self.opt.att_mode!='vis_concat_attimg_gate':
                                    t_outputs = self.model(t_inputs, t_imgs_f, t_img_mean, t_img_att, self.opt.att_mode)
                                else:
                                    t_outputs,_,_,_ = self.model(t_inputs, t_imgs_f, t_img_mean, t_img_att, self.opt.att_mode)

                                n_dev_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                                n_dev_total += len(t_outputs)

                                true_label_list.append(t_targets.detach().cpu())
                                pred_label_list.append(t_outputs.detach().cpu())

                                # if true_label is None:
                                #     true_label = t_targets.detach().cpu()
                                #     pred_outputs = t_outputs.detach().cpu()
                                # else:
                                #     true_label = np.concatenate((true_label, t_targets.detach().cpu()))
                                #     pred_outputs = np.concatenate((pred_outputs, t_outputs.detach().cpu()))

                            true_label = np.concatenate(true_label_list)
                            pred_outputs = np.concatenate(pred_label_list)
                                
                            dev_p, dev_r, dev_f1 = macro_f1(true_label, pred_outputs)
                            dev_acc = n_dev_correct / n_dev_total

                            # true_label = None
                            # pred_outputs = None
                            true_label_list = []
                            pred_label_list = []
                            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                                t_inputs = [t_sample_batched[col].to(device) for col in self.opt.inputs_cols]
                                t_targets = t_sample_batched['polarity'].to(device)
                                t_images = t_sample_batched['image'].to(device)
                            
                                t_imgs_f, t_img_mean, t_img_att = self.encoder(t_images)
                            
                                if self.opt.att_mode!='vis_att_img_gate' and self.opt.att_mode!='vis_concat_attimg_gate':
                                    t_outputs = self.model(t_inputs, t_imgs_f, t_img_mean, t_img_att, self.opt.att_mode)
                                else:
                                    t_outputs,_,_,_ = self.model(t_inputs, t_imgs_f, t_img_mean, t_img_att, self.opt.att_mode)

                                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                                n_test_total += len(t_outputs)

                                true_label_list.append(t_targets.detach().cpu())
                                pred_label_list.append(t_outputs.detach().cpu())

                                # if true_label is None:
                                #     true_label = t_targets.detach().cpu()
                                #     pred_outputs = t_outputs.detach().cpu()
                                # else:
                                #     true_label = np.concatenate((true_label, t_targets.detach().cpu()))
                                #     pred_outputs = np.concatenate((pred_outputs, t_outputs.detach().cpu()))

                            true_label = np.concatenate(true_label_list)
                            pred_outputs = np.concatenate(pred_label_list)
                            
                            test_p, test_r, test_f1  = macro_f1(true_label, pred_outputs)
                            test_acc = n_test_correct / n_test_total
                            #if test_acc > max_test_acc:
                            # if dev_acc > max_dev_acc:
                            # if dev_acc+dev_f1 > max_dev_acc+max_dev_f1:
                            if dev_acc > max_dev_acc:
                                max_dev_acc = dev_acc
                                max_dev_p = dev_p
                                max_dev_r = dev_r
                                max_dev_f1 = dev_f1
                                max_test_acc = test_acc
                                max_test_p = test_p
                                max_test_r = test_r
                                max_test_f1 = test_f1
                                
                                track_list.append(
                                    {'loss': loss.item(), 'dev_acc': dev_acc, 'dev_p': dev_p,'dev_r': dev_r,'dev_f1': dev_f1, \
                                     'test_acc': test_acc, 'test_p': test_p,'test_r': test_r,'test_f1': test_f1})
                            
                                save_checkpoint({'epoch': epoch,
                                                    'state_dict': self.model.state_dict(),
                                                    'encode_dict': self.encoder.state_dict(),
                                                    'optimizer': optimizer.state_dict(),
                                                }, {'track_list': track_list}, 
                                                self.opt.checkpoint + self.opt.dataset+ '_'+ self.opt.model_name+ '_'+ self.opt.att_mode)

                            print('loss: {:.4f}, acc: {:.4f}, dev_acc: {:.4f}, test_acc: {:.4f}'.format(loss.item(),\
                                   train_acc, dev_acc, test_acc))
                               
                            print('dev_f1: {:.4f}, test_f1: {:.4f}'.format(dev_f1, test_f1))
                            # log
                            #self.writer.add_scalar('loss', loss, global_step)
                            #self.writer.add_scalar('acc', train_acc, global_step)
                            #self.writer.add_scalar('test_acc', test_acc, global_step)

            #self.writer.close()

            print('max_dev_acc: {0}, test_acc: {1}'.format(max_dev_acc, max_test_acc))
            print('dev_p: {0}, dev_r: {1}, dev_f1: {2}, test_p: {3}, test_r: {4}, test_f1: {5}'.format(max_dev_p,\
                     max_dev_r, max_dev_f1, max_test_p, max_test_r, max_test_f1))
            
if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand_seed', default=8, type=int)
    parser.add_argument('--model_name', default='mmtdtan', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, snap')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--num_epoch', default=8, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--log_step', default=50, type=int) # 50
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--max_seq_len', default=64, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--att_file', default='./att_file/', help='path of attention file')
    parser.add_argument('--pred_file', default='./pred_file/', help='path of prediction file')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='grad clip at')
    #parser.add_argument('--path_image', default='../../../visual_attention_ner/twitter_subimages', help='path to images')
    parser.add_argument('--path_image', default='./twitter_subimages', help='path to images')
    #parser.add_argument('--path_image', default='./twitter15_images', help='path to images')
    #parser.add_argument('--path_image', default='./snap_subimages', help='path to images')
    parser.add_argument('--crop_size', type=int, default = 224, help='crop size of image')
    parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
    parser.add_argument('--att_mode', choices=['text', 'vis_only', 'vis_concat',  'vis_att', 'vis_concat_attimg', \
                                               'text_vis_att_img_gate', 'vis_att_concat', 'vis_att_attimg', \
    'vis_att_img_gate', 'vis_concat_attimg_gate'], default ='vis_concat_attimg_gate', \
    help='different attention mechanism')
    parser.add_argument('--resnet_root', default='./resnet', help='path the pre-trained cnn models')
    parser.add_argument('--checkpoint', default='./checkpoint/', help='path to checkpoint prefix')
    parser.add_argument('--load_check_point', action='store_true', help='path of checkpoint')
    parser.add_argument('--load_opt', action='store_true', help='load optimizer from ')
    parser.add_argument('--tfn', action='store_true', help='whether to use TFN')
    
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if opt.dataset == "twitter":
        opt.path_image = "../multi_modal_ABSA_pytorch_bilinear/twitter_subimages/"
        opt.max_seq_len = 27
        opt.rand_seed = 28
    elif opt.dataset == "twitter2015":
        opt.path_image = "../multi_modal_ABSA_pytorch_bilinear/twitter15_images/"
        opt.max_seq_len = 24
        opt.rand_seed = 25
    else:
        print("The task name is not right!")

    if opt.tfn:
        print("************add another tfn layer*************")
    else:
        print("************no tfn layer************")

    random.seed(opt.rand_seed)
    np.random.seed(opt.rand_seed)
    torch.manual_seed(opt.rand_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(opt.rand_seed)

    model_classes = {
        'mmian': MMIAN,
        'mmram': MMRAM,
        'mmmgan': MMMGAN,
        'mmfusion': MMFUSION
    }
    input_colses = {
        'mmian': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'mmram': ['text_raw_indices', 'aspect_indices'],
        'mmmgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'mmfusion': ['text_left_indicator', 'text_right_indicator', 'aspect_indices']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = device

    ins = Instructor(opt)
    ins.run()
