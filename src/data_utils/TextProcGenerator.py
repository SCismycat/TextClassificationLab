#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/2/2 10:13
# @Author  : Leslee
import os
import numpy as np

from src.data_utils.TextProcUtil import load_json, save_json


class DataGenerator(object):

    def __init__(self, path_model_dir):
        self.label2idx_idx2label = None
        self.path_fast_text_model_vocab2index = path_model_dir + 'vocab2index.json'
        self.path_fast_text_model_label2idx_idx2label = path_model_dir + 'l2i_i2l.json'
        if os.path.exists(self.path_fast_text_model_label2idx_idx2label):
            self.label2idx_idx2label = load_json(self.path_fast_text_model_label2idx_idx2label)

    def preprocess_idx(self, pred):
        if os.path.exists(self.path_fast_text_model_label2idx_idx2label):
            pred_idx2label = {}
            index2label = self.label2idx_idx2label['i2l']
            for i in range(len(pred)):
                pred_idx2label[index2label[str(i)]] = pred[i]
            pred_idx2label_rank = [sorted(pred_idx2label.items(), key=lambda k:k[1], reverse=True)]
            return pred_idx2label_rank
        else:
            raise RuntimeError("path_fast_text_model_label2index is None")

    def preprocess_pred_xid(self, pred):
        if os.path.exists(self.path_fast_text_model_label2idx_idx2label):
            pred_label2idx = {}
            label2idx = self.label2idx_idx2label['l2i']
            for i in range(len(pred)):
                pred_label2idx[pred[i]] = label2idx[pred[i]]
            pred_label2idx_rank = [sorted(pred_label2idx.items(), key=lambda k: k[1], reverse=True)]
            return pred_label2idx_rank
        else:
            raise RuntimeError("path_fast_text_model_label2index is None")

    def preprocess_get_label_set(self, path):
        label_set = set()
        len_all = 0
        file_csv = open(path, 'r', encoding='utf-8')
        for line in file_csv:
            len_all += 1
            if len_all > 1:   # 第一条是标签'label,ques'，不选择
                line_sp = line.split(",")
                label = str(line_sp[0]).strip().upper()
                label_real = "NAN" if label == "" else label
                label_set.add(label_real)
        file_csv.close()
        return label_set, len_all

    def preprocess_label_ques_to_idx(self, embedding_type, batch_size, path, embed, rate=1, epoch=20):
        label_set, len_all = self.preprocess_get_label_set(path)
        if not os.path.exists(self.path_fast_text_model_label2idx_idx2label):
            count = 0
            label2idx = {}
            idx2label = {}
            for label_one in label_set:
                label2idx[label_one] = count
                idx2label[count] = label_one
                count = count + 1
            label2idxandidx2label = dict()
            label2idxandidx2label['l2i'] = label2idx
            label2idxandidx2label['i2l'] = idx2label
            save_json(label2idxandidx2label, self.path_fast_text_model_label2idx_idx2label)
        else:
            label2idxandidx2label = load_json(self.path_fast_text_model_label2idx_idx2label)

        len_rate = int(rate * len_all)
        if len_rate <= 500:
            len_rate = len_rate

        def process_line(line):
            # 针对每条数据获取label和句子的索引
            line_sp = line.split(",")
            ques = str(line_sp[1]).strip().upper()
            label = str(line_sp[0]).strip().upper()
            label = "NAN" if label == "" else label
            ques_embed = embed.sentence2idx(ques)
            # 这里是对label在进行one-hot，Todo：转为使用keras进行one_hot
            label_zeros = [0] * len(label2idxandidx2label['l2i'])
            label_zeros[label2idxandidx2label['l2i'][label]] = 1
            return ques_embed, label_zeros
        for _ in range(epoch):
            while True:
                file_csv = open(path, "r", encoding='utf-8')
                count_all_lines = 0
                cnt = 0
                x, y = [], []
                if len_rate < count_all_lines:
                    break
                for line in file_csv:
                    count_all_lines += 1
                    if count_all_lines > 1:  # 如果第一行是表头，则不读表头，todo：可能会遗漏数据
                        x_line, y_line = process_line(line)
                        x.append(x_line)
                        y.append(y_line)
                        cnt += 1
                        # 这里是在组装batch的数据，todo：看看keras组装batch的原生方法。
                        if cnt == batch_size:
                            if embedding_type in ['bert', 'albert']:
                                x_, y_ = np.array(x), np.array(y)
                                x_1 = np.array([x[0] for x in x_])
                                x_2 = np.array([x[1] for x in x_])
                                x_all = [x_1, x_2]
                            elif embedding_type == 'xlnet':
                                x_, y_ = x, np.array(y)
                                x_1 = np.array([x[0][0] for x in x_])
                                x_2 = np.array([x[1][0] for x in x_])
                                x_3 = np.array([x[2][0] for x in x_])
                                x_all = [x_1, x_2, x_3]
                            else:
                                x_all, y_ = np.array(x), np.array(y)
                            cnt = 0
                            yield (x_all, y_)
                            x, y = [], []
            file_csv.close()
        print("preprocess_label_ques_to_idx ok")



