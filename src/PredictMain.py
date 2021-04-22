#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/2/5 15:21
# @Author  : Leslee
import os
import sys
import pathlib
import time
import numpy as np
from sklearn.metrics import classification_report

from src.model.model_repo.BertModel import BertModel
from src.data_utils.TextProcUtil import PreprocessText, read_and_process, load_json
from src.conf.path_config import data_path_file_train, data_path_file_valid, path_model_dir, path_hyper_parameters


def pred_test(path_hyper_parameters=path_hyper_parameters, path_test=None, rate=1.0):

    hyper_parameters = load_json(path_hyper_parameters)
    if path_test:
        hyper_parameters['data']['val_data'] = path_test
    time_start = time.time()
    graph = BertModel(hyper_parameters)
    print("graph init ok!")
    graph.load_model()
    print("graph load ok!")
    input_embedding = graph.word_embedding
    process_text = PreprocessText(path_model_dir)
    y, x = read_and_process(hyper_parameters['data']['val_data'])
    len_rate = int(len(y) * rate)
    x = x[1: len_rate]
    y = y[1: len_rate]
    y_pred = []
    count = 0
    for one in x:
        count += 1
        ques_embed = input_embedding.sentence2idx(one)
        if hyper_parameters['embedding_type'] in ['bert', 'albert']:  # bert数据处理, token
            x_val_1 = np.array([ques_embed[0]])
            x_val_2 = np.array([ques_embed[1]])
            x_val = [x_val_1, x_val_2]
        else:
            x_val = ques_embed
        pred = graph.predict(x_val)
        true_label = process_text.prereocess_idx(pred[0])
        label_pred = true_label[0][0][0]
        if count % 1000 == 0:
            print(label_pred)
        y_pred.append(label_pred)
    # 预测
    print("data pred ok!")
    index_y = [process_text.l2i_i2l['l2i'][i] for i in y]
    index_pred = [process_text.l2i_i2l['l2i'][i] for i in y_pred]
    target_names = [process_text.l2i_i2l['i2l'][str(i)] for i in list(set((index_pred + index_y)))]
    # 评估
    report_predict = classification_report(index_y, index_pred, target_names=target_names, digits=9)
    print(report_predict)
    print("耗时:" + str(time.time() - time_start))


def model_predict(input_data):

    hyper_parameters = load_json(path_hyper_parameters)
    process_text = PreprocessText(path_model_dir)

    # 模式初始化和加载
    graph = BertModel(hyper_parameters)
    graph.load_model()
    input_embedding = graph.word_embedding

    if isinstance(input_data, str):
        result_list = []
        ques_embed = input_embedding.sentence2idx(input_data)
        if hyper_parameters['embedding_type'] in ['bert', 'albert']:
            x_val_1 = np.array([ques_embed[0]])
            x_val_2 = np.array([ques_embed[1]])
            x_val = [x_val_1, x_val_2]
        else:
            x_val = ques_embed
        pred_result = {}
        pred = graph.predict(x_val)
        pre = process_text.prereocess_idx(pred[0])
        pred_result["text"] = input_data
        pred_result["label"] = pre
        result_list.append(pred_result)
        return result_list
    elif isinstance(input_data, list):
        result_list = []
        for data in input_data:
            pred_dict = {}
            ques_embed = input_embedding.sentence2idx(data)
            if hyper_parameters['embedding_type'] in ['bert', 'albert']:
                x_val_1 = np.array([ques_embed[0]])
                x_val_2 = np.array([ques_embed[1]])
                x_val = [x_val_1, x_val_2]
            else:
                x_val = ques_embed
            pred = graph.predict(x_val)
            pre = process_text.prereocess_idx(pred[0])
            pred_dict["text"] = data
            pred_dict["label"] = pre
            result_list.append(pred_dict)
            return result_list


if __name__ == '__main__':

    # pred_test(path_test=data_path_file_valid, rate=1)  # sample条件下设为1,否则训练语料可能会很少
    while True:
        input_data = input("Please Enter Text:")
        result = model_predict(input_data)
        print(result)
