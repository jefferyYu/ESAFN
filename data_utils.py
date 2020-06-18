# -*- coding: utf-8 -*-
# file: data_utils.py
# author: jyu5 <yujianfei1990@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec

def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.dat'.format(str(embed_dim), type)
    load_embedding_boolean = True
    if load_embedding_boolean:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        #fname = '/home/jfyu/torch/stanford_treelstm-master/data/glove/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            #if embed_dim != 300 else '/home/jfyu/torch/stanford_treelstm-master/data/glove/glove.840B.300d.txt'
        fname = '../../../pytorch/glove.twitter.27B.' + str(embed_dim) + 'd.txt'
            #if embed_dim != 200 else '../../../pytorch/glove.6B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

class Tokenizer(object):
    def __init__(self, lower=False, max_seq_len=None, max_aspect_len=None):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.max_aspect_len = max_aspect_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 2
        self.word2idx['ttttt'] = 1
        self.idx2word[1] = 'ttttt'

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='pre', truncating='pre', value=0.):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = 'post'  # use post padding together with torch.nn.utils.rnn.pack_padded_sequence
        if reverse:
            sequence = sequence[::-1]
        return Tokenizer.pad_sequence(sequence, self.max_seq_len, dtype='int64', padding=pad_and_trunc, truncating=pad_and_trunc)


class ABSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 4):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer, path_img, transform):
        print('--------------'+fname+'---------------')
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        count = 0
        for i in range(0, len(lines), 4):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            image_id = lines[i + 3].strip()
            
            if text_left == "":
                text_left_for_tdan = "$unk$"
            else:
                text_left_for_tdan = text_left
            if text_right == "":
                text_right_for_tdan = "$unk$"
            else:
                text_right_for_tdan = text_right

            text_left_for_fusion = text_left + " ttttt"
            text_right_for_fusion = "ttttt " + text_right

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            if text_left == "" and text_right == "":
                text_raw_without_aspect_indices = tokenizer.text_to_sequence("$unk$")
            else:
                text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left_for_tdan)
            text_left_indicator = tokenizer.text_to_sequence(text_left_for_fusion)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right_for_tdan, reverse=True)
            text_right_indicator = tokenizer.text_to_sequence(text_right_for_fusion)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            polarity = int(polarity)+1
            
            image_name = image_id
            image_path = os.path.join(path_img, image_name)
        
            if not os.path.exists(image_path):
                print(image_path)
            try:
                image = image_process(image_path, transform)
            except:
                count += 1
                #print('image has problem!')
                image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
                image = image_process(image_path_fail, transform)

            data = {
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_indicator': text_left_indicator,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_indicator': text_right_indicator,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
                'image': image,
            }

            all_data.append(data)
        print('the number of problematic samples: '+str(count))
        return all_data

    def __init__(self, transform, dataset='twitter', embed_dim=100, max_seq_len=40, path_image='./twitter_subimages'):
        print("preparing {0} dataset...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/twitter/train.txt',
                'dev': './datasets/twitter/dev.txt',
                'test': './datasets/twitter/test.txt'
            },
            'twitter2015': {
                'train': './datasets/twitter2015/train.txt',
                'dev': './datasets/twitter2015/dev.txt',
                'test': './datasets/twitter2015/test.txt'
            },
            'snap': {
                'train': './datasets/snap/train.txt',
                'dev': './datasets/snap/dev.txt',
                'test': './datasets/snap/test.txt'
            }
        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['dev'], fname[dataset]['test']])
        tokenizer = Tokenizer(max_seq_len=max_seq_len)
        tokenizer.fit_on_text(text.lower())
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer, path_image, transform))
        self.dev_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['dev'], tokenizer, path_image, transform))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer, path_image, transform))
