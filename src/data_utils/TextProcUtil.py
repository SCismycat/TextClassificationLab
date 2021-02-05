#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/1/30 15:25
# @Author  : Leslee
import re
import os
import json
import jieba
import random
import pandas as pd
import numpy as np
from tqdm import tqdm


__all__ = ["PreprocessText", "PreprocessTextMulti", "PreprocessSim"]
__tools__ = ["txt_read", "txt_write", "extract_chinese", "read_and_process",
             "preprocess_label_ques", "save_json", "load_json", "delete_file",
             "transform_multilabel_to_multihot"]


def txt_read(file_path, encode_type='utf-8'):
    """
      读取txt文件，默认utf8格式
    :param file_path: str, 文件路径
    :param encode_type: str, 编码格式
    :return: list
    """
    list_line = []
    try:
        file = open(file_path, 'r', encoding=encode_type)
        while True:
            line = file.readline()
            line = line.strip()
            if not line:
                break
            list_line.append(line)
        file.close()
    except Exception as e:
        print(str(e))
    finally:
        return list_line


def get_ngram(text, ns=[1]):

    if type(ns) != list:
        raise RuntimeError("ns of function get_ngram() must be list!")
    for n in ns:
        if n < 1:
            raise RuntimeError("enum of ns must '>1'!")
    len_text = len(text)
    ngrams = []
    for n in ns:
        ngram_n = []
        for i in range(len_text):
            if i + n <= len_text:
                ngram_n.append(text[i:i+n])
            else:
                break
        if not ngram_n:
            ngram_n.append(text)
        ngrams += ngram_n
    return ngrams


def txt_write(list_line, file_path, type='w', encode_type='utf-8'):
    """
      txt写入list文件
    :param listLine:list, list文件，写入要带"\n"
    :param filePath:str, 写入文件的路径
    :param type: str, 写入类型, w, a等
    :param encode_type:
    :return:
    """
    try:
        file = open(file_path, type, encoding=encode_type)
        file.writelines(list_line)
        file.close()

    except Exception as e:
        print(str(e))


def extract_chinese(text):
    """
      只提取出中文、字母和数字
    :param text: str, input of sentence
    :return:
    """
    chinese_exttract = ''.join(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@._])", text))
    return chinese_exttract


def read_and_process(path):
    """
      读取文本数据并
    :param path:
    :return:
    """
    # with open(path, 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     line_x = [extract_chinese(str(line.split(",")[0])) for line in lines]
    #     line_y = [extract_chinese(str(line.split(",")[1])) for line in lines]
    #     return line_x, line_y

    data = pd.read_csv(path)
    ques = data["ques"].values.tolist()
    labels = data["label"].values.tolist()
    line_x = [extract_chinese(str(line).upper()) for line in labels]
    line_y = [extract_chinese(str(line).upper()) for line in ques]
    return line_x, line_y


def preprocess_label_ques(path):
    x, y, x_y = [], [], []
    x_y.append('label,ques\n')
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            try:
                line_json = json.loads(line)
            except:
                break
            ques = line_json['title']
            label = line_json['category'][0:2]
            line_x = " ".join([extract_chinese(word) for word in list(jieba.cut(ques, cut_all=False, HMM=True))]).strip().replace('  ',' ')
            line_y = extract_chinese(label)
            x_y.append(line_y+','+line_x+'\n')
    return x_y


def load_json(path):
    """
      获取json，只取第一行
    :param path: str
    :return: json
    """
    with open(path, 'r', encoding='utf-8') as fj:
        model_json = json.loads(fj.readlines()[0])
    return model_json


def save_json(jsons, json_path):
    """
      保存json，
    :param json_: json
    :param path: str
    :return: None
    """
    with open(json_path, 'w', encoding='utf-8') as fj:
        fj.write(json.dumps(jsons, ensure_ascii=False))
    fj.close()


def delete_file(path):
    """
        删除一个目录下的所有文件
    :param path: str, dir path
    :return: None
    """
    if not os.path.exists(path):
        return
    for i in os.listdir(path):
        # 取文件或者目录的绝对路径
        path_children = os.path.join(path, i)
        if os.path.isfile(path_children):
            if path_children.endswith(".h5") or path_children.endswith(".json"):
                os.remove(path_children)
        else:# 递归, 删除目录下的所有文件
            delete_file(path_children)


def transform_multilabel_to_multihot(sample, label=1070):
    result = np.zeros(label)
    result[sample] = 1
    res = result.tolist()
    return res


class PreprocessText:
    """
        数据预处理, 输入为csv格式, [label,ques]
    """
    def __init__(self, path_model_dir):
        self.l2i_i2l = None
        self.path_fast_text_model_vocab2index = path_model_dir + 'vocab2index.json'
        self.path_fast_text_model_l2i_i2l = path_model_dir + 'l2i_i2l.json'
        if os.path.exists(self.path_fast_text_model_l2i_i2l):
            self.l2i_i2l = load_json(self.path_fast_text_model_l2i_i2l)

    def prereocess_idx(self, pred, digits=5):
        if os.path.exists(self.path_fast_text_model_l2i_i2l):
            pred_i2l = {}
            i2l = self.l2i_i2l['i2l']
            for i in range(len(pred)):
                pred_i2l[i2l[str(i)]] = round(float(pred[i]), digits)
            pred_i2l_rank = [sorted(pred_i2l.items(), key=lambda k: k[1], reverse=True)]
            return pred_i2l_rank
        else:
            raise RuntimeError("path_fast_text_model_label2index is None")

    def prereocess_pred_xid(self, pred):
        if os.path.exists(self.path_fast_text_model_l2i_i2l):
            pred_l2i = {}
            l2i = self.l2i_i2l['l2i']
            for i in range(len(pred)):
                pred_l2i[pred[i]] = l2i[pred[i]]
            pred_l2i_rank = [sorted(pred_l2i.items(), key=lambda k: k[1], reverse=True)]
            return pred_l2i_rank
        else:
            raise RuntimeError("path_fast_text_model_label2index is None")

    def preprocess_label_ques_to_idx(self, embedding_type, path, embed, rate=1, shuffle=True, graph=None):
        data = pd.read_csv(path, quoting=3)
        ques = data['ques'].tolist()
        label = data['label'].tolist()
        ques = [str(q).upper() for q in ques]
        label = [str(l).upper() for l in label]
        if shuffle:
            ques = np.array(ques)
            label = np.array(label)
            indexs = [ids for ids in range(len(label))]
            random.shuffle(indexs)
            ques, label = ques[indexs].tolist(), label[indexs].tolist()
        # 如果label2index存在则不转换了
        if not os.path.exists(self.path_fast_text_model_l2i_i2l):
            label_set = set(label)
            count = 0
            label2index = {}
            index2label = {}
            for label_one in label_set:
                label2index[label_one] = count
                index2label[count] = label_one
                count = count + 1

            l2i_i2l = {}
            l2i_i2l['l2i'] = label2index
            l2i_i2l['i2l'] = index2label
            save_json(l2i_i2l, self.path_fast_text_model_l2i_i2l)
        else:
            l2i_i2l = load_json(self.path_fast_text_model_l2i_i2l)

        len_ql = int(rate * len(ques))
        if len_ql <= 500: # sample时候不生效,使得语料足够训练
            len_ql = len(ques)

        x = []
        print("ques to index start!")
        ques_len_ql = ques[0:len_ql]
        for i in tqdm(range(len_ql)):
            que = ques_len_ql[i]
            que_embed = embed.sentence2idx(que)
            x.append(que_embed) # [[], ]
        label_zo = []
        print("label to onehot start!")
        label_len_ql = label[0:len_ql]
        for j in tqdm(range(len_ql)):
            label_one = label_len_ql[j]
            label_zeros = [0] * len(l2i_i2l['l2i'])
            label_zeros[l2i_i2l['l2i'][label_one]] = 1
            label_zo.append(label_zeros)

        count = 0
        if embedding_type in ['bert', 'albert']:
            x_, y_ = np.array(x), np.array(label_zo)
            x_1 = np.array([x[0] for x in x_])
            x_2 = np.array([x[1] for x in x_])
            x_all = [x_1, x_2]
            return x_all, y_
        elif embedding_type == 'xlnet':
            count += 1
            if count == 1:
                x_0 = x[0]
                print(x[0][0][0])
            x_, y_ = x, np.array(label_zo)
            x_1 = np.array([x[0][0] for x in x_])
            x_2 = np.array([x[1][0] for x in x_])
            x_3 = np.array([x[2][0] for x in x_])
            if embed.trainable:
                x_4 = np.array([x[3][0] for x in x_])
                x_all = [x_1, x_2, x_3, x_4]
            else:
                x_all = [x_1, x_2, x_3]
            return x_all, y_
        else:
            x_, y_ = np.array(x), np.array(label_zo)
            return x_, y_

