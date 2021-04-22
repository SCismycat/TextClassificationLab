#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/2/6 11:01
# @Author  : Leslee
from __future__ import absolute_import
import json
import sys
import time
import logging
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from tensorflow.python.keras.backend import set_session

from src.model.model_repo.BertModel import BertModel
from src.utils.return_format import my_result_format
from src.data_utils.TextProcUtil import PreprocessText, load_json
from src.conf.path_config import path_model_dir, path_hyper_parameters
from src.pyapollo.apollo import get_config, stop_apollo
# 配置项
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
# 读取apollo
# port = get_config("classify_port", "textclassify")
# stop_apollo()
port = 8080
# 初始化加载字典、模型等需要加载的文件
logging.info("Loading Label Data And Files..")
try:
    hyper_parameters = load_json(path_hyper_parameters)
    process_text = PreprocessText(path_model_dir)
except Exception as e:
    logging.error("error when load the hyper_params, Please check the model path and hyper path", e)
    sys.exit(1)
logging.info("Loaded Label Data And Files!")
sess = tf.Session()
graph = tf.get_default_graph()
# 模式初始化和加载
logging.info("Loading Bert Model..")
try:
    set_session(sess)
    model = BertModel(hyper_parameters)
    model.load_model()
    input_embedding = model.word_embedding
except Exception as e:
    logging.error("error when loading Model, Please check the model path", e)
    sys.exit(1)
logging.info("Bert Model Loaded!")


@app.route("/api/nlplab/textclassify", methods=["POST"])
def textclassify():
    inputs = request.get_data()
    inputs_data = json.loads(inputs, encoding="utf-8")
    contents = inputs_data.get("input")
    start_time = time.time()
    if isinstance(contents, str):
        logging.info("Service with input String")
        ques_embed = input_embedding.sentence2idx(contents)
        if hyper_parameters['embedding_type'] in ['bert', 'albert']:
            x_val_1 = np.array([ques_embed[0]])
            x_val_2 = np.array([ques_embed[1]])
            x_val = [x_val_1, x_val_2]
        else:
            x_val = ques_embed
        logging.info("Start Predict!")
        with graph.as_default():
            pred = model.predict(x_val)
        pre = process_text.prereocess_idx(pred[0])
        pd_classify = []  # 所有的标签和分数
        if isinstance(pre, list) and len(pre) >= 1:
            logging.info("parser the predict result!")
            for label_score in pre[0]:
                label_dict = dict()
                label = label_score[0]
                score = label_score[1]
                label_dict["label"] = label
                label_dict["score"] = score
                pd_classify.append(label_dict)
        else:
            logging.warning("Failed to Get the predict result!")
        result = my_result_format(contents, pd_classify)
        end_time = time.time()
        cost_time = end_time - start_time
        logging.info("Cost Time: %s" % cost_time)
        final_result = {
            "data": result,
            "errCode": 200,
            "message": ""
        }
        return jsonify(final_result)
    elif isinstance(contents, list):
        logging.info("Service with input List")
        final_results = []
        for data in contents:
            ques_embed = input_embedding.sentence2idx(data)
            if hyper_parameters['embedding_type'] in ['bert', 'albert']:
                x_val_1 = np.array([ques_embed[0]])
                x_val_2 = np.array([ques_embed[1]])
                x_val = [x_val_1, x_val_2]
            else:
                x_val = ques_embed
            with graph.as_default():
                pred = model.predict(x_val)
            pre = process_text.prereocess_idx(pred[0])
            pd_classify = []
            if isinstance(pre, list) and len(pre) >= 1:
                logging.info("parser the predict result!")
                for label_score in pre[0]:
                    label_dict = dict()
                    label = label_score[0]
                    score = label_score[1]
                    label_dict["label"] = label
                    label_dict["score"] = score
                    pd_classify.append(label_dict)
            else:
                logging.warning("Failed to Get the predict result!")
            final_result = my_result_format(data, pd_classify)
            final_results.append(final_result)
        last_results = {
            "data": final_results,
            "errCode": 200,
            "message": ""
        }
        return jsonify(last_results)
    else:
        raise Exception("Please give current params")


# 健康检查接口
@app.route("/actuator/health", methods=["GET"])
def health_check():
    status = dict()
    status["status"] = "UP"
    return jsonify(status)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
