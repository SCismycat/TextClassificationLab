#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 10:13
# @Author  : Leslee
import os
import logging
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from src.keras_layers.keras_radam import RAdam
from src.keras_layers.keras_lookahead import Lookahead
from src.data_utils.TextProcUtil import save_json
from src.data_utils.TextProcGenerator import DataGenerator
from src.conf.path_config import path_model, path_fineture, path_model_dir, path_hyper_parameters


class Graph:

    def __init__(self, hyper_parameters):
        self.len_max = hyper_parameters.get('len_max', 50)  # 文本最大长度
        self.embed_size = hyper_parameters.get('embed_size', 300)
        self.trainable = hyper_parameters.get('trainable', False)
        self.embedding_type = hyper_parameters.get('embedding_type', 'word2vec')
        self.gpu_memory_fraction = hyper_parameters.get('gpu_memory_fraction', None)
        self.hyper_parameters = hyper_parameters
        hyper_parameters_model = hyper_parameters['model']
        self.label = hyper_parameters_model.get('label', 2)
        self.batch_size = hyper_parameters_model.get('batch_size', 32)
        self.filters = hyper_parameters_model.get('filters', [3, 4, 5])
        self.filters_num = hyper_parameters_model.get('filters_num', 300)
        self.channel_size = hyper_parameters_model.get('channel_size', 1)
        self.dropout = hyper_parameters_model.get('dropout', 0.5)
        self.decay_step = hyper_parameters_model.get('decay_step', 100)
        self.decay_rate = hyper_parameters_model.get('decay_rate', 0.9)
        self.epochs = hyper_parameters_model.get('epochs', 20)
        self.vocab_size = hyper_parameters_model.get('vocab_size', 20000)
        self.lr = hyper_parameters_model.get('lr', 1e-3)
        self.l2 = hyper_parameters_model.get('l2', 1e-6)
        self.activate_classify = hyper_parameters_model.get('activate_classify', 'softmax')
        self.loss = hyper_parameters_model.get('loss', 'categorical_crossentropy')
        self.metrics = hyper_parameters_model.get('metrics', 'accuracy')
        self.is_training = hyper_parameters_model.get('is_training', False)
        self.path_model_dir = hyper_parameters_model.get('path_model_dir', path_model_dir)
        self.model_path = hyper_parameters_model.get('model_path', path_model)
        self.path_hyper_parameters = hyper_parameters_model.get('path_hyper_parameters', path_hyper_parameters)
        self.path_fineture = hyper_parameters_model.get('path_fineture', path_fineture)
        self.patience = hyper_parameters_model.get('patience', 3)
        self.optimizer_name = hyper_parameters_model.get('optimizer_name', 'Adam')
        if self.gpu_memory_fraction:
            import tensorflow as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            K.set_session(sess)
        self.create_model(hyper_parameters)
        if self.is_training:
            self.create_compile()

    def create_model(self, hyper_parameters):

        Embeddings = None
        if self.embedding_type == 'random':
            from src.Embedding.embedding import RandomEmbedding as Embeddings
        elif self.embedding_type == 'bert':
            from src.Embedding.embedding import BertEmbedding as Embeddings
        elif self.embedding_type == 'xlnet':
            from src.Embedding.embedding import XlnetEmbedding as Embeddings
        elif self.embedding_type == 'albert':
            from src.Embedding.embedding import AlbertEmbedding as Embeddings
        elif self.embedding_type == 'word2vec':
            from src.Embedding.embedding import WordEmbedding as Embeddings
        else:
            raise RuntimeError("Input must be 'xlnet'、'random'、 'bert'、 'albert' or 'word2vec")
        self.word_embedding = Embeddings(hyper_parameters=hyper_parameters)
        if os.path.exists(self.path_fineture) and self.trainable:
            self.word_embedding.model.load_weights(self.path_fineture)
            print("load path_finetune OK!")
        self.model = None

    def callback(self):
        cb_em = [TensorBoard(log_dir=os.path.join(self.path_model_dir, "logs"), batch_size=self.batch_size,
                             update_freq='batch'),
                 EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-8, patience=self.patience),
                 ModelCheckpoint(monitor='val_loss', mode='min', filepath=self.model_path, verbose=1,
                                 save_best_only=True, save_weights_only=True)]
        return cb_em

    def create_compile(self):

        if self.optimizer_name.upper() == 'ADAM':
            self.model.compile(optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                               loss=self.loss, metrics=[self.metrics])
        elif self.optimizer_name.upper() == 'RADAM':
            self.model.compile(optimizer=RAdam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                               loss=self.loss, metrics=[self.metrics])
        else:
            self.model.compile(optimizer=RAdam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                               loss=self.loss, metrics=[self.metrics])
            lookahead = Lookahead(k=5, alpha=0.5)
            lookahead.inject(self.model)

    def fit(self, x_train, y_train, x_dev, y_dev):
        # 当不是训练状态的时候，需要设置以下参数，Todo：但是这里是训练呀？
        self.hyper_parameters['model']['is_training'] = False
        self.hyper_parameters['model']['trainable'] = False
        self.hyper_parameters['model']['dropout'] = 0.0
        save_json(jsons=self.hyper_parameters, json_path=self.path_hyper_parameters)

        self.model.fit(x_train, y_train, batch_size=self.batch_size,
                       epochs=self.epochs, validation_data=(x_dev, y_dev),
                       shuffle=True, callbacks=self.callback())
        if self.trainable:
            self.word_embedding.model.save(self.path_fineture)

    def fit_generator(self, embed, rate=1):

        # 保存超参数
        self.hyper_parameters['model']['is_training'] = False  # 预测时候这些设为False
        self.hyper_parameters['model']['trainable'] = False
        self.hyper_parameters['model']['dropout'] = 0.0
        save_json(jsons=self.hyper_parameters, json_path=self.path_hyper_parameters)

        data_gene = DataGenerator(self.path_model_dir)
        _, len_train = data_gene.preprocess_get_label_set(self.hyper_parameters['data']['train_data'])
        data_fit_generator = data_gene.preprocess_label_ques_to_idx(embedding_type=
                                                                    self.hyper_parameters['embedding_type'],
                                                                    batch_size=self.batch_size,
                                                                    path=self.hyper_parameters['data']['train_data'],
                                                                    embed=embed,
                                                                    epoch=self.epochs,
                                                                    rate=rate)
        _, len_val = data_gene.preprocess_get_label_set(self.hyper_parameters['data']['val_data'])
        data_dev_generator = data_gene.preprocess_label_ques_to_idx(embedding_type=
                                                                    self.hyper_parameters['embedding_type'],
                                                                    batch_size=self.batch_size,
                                                                    path=self.hyper_parameters['data']['val_data'],
                                                                    embed=embed,
                                                                    epoch=self.epochs,
                                                                    rate=rate)
        steps_per_epoch = len_train // self.batch_size + 1
        validation_steps = len_val // self.batch_size + 1
        self.model.fit_generator(generator=data_fit_generator,
                                 validation_data=data_dev_generator,
                                 callbacks=self.callback(),
                                 epochs=self.epochs,
                                 steps_per_epoch=32,
                                 validation_steps=6)
        if self.trainable:
            self.word_embedding.model.save(self.path_fineture)

    def load_model(self):

        logging.info("load model Start!")
        self.model.load_weights(self.model_path)
        logging.info("Loaded Model!")

    def predict(self, sentence):
        if self.embedding_type in ['bert', 'xlnet', 'albert']:
            if type(sentence) == np.ndarray:
                sentence = sentence.tolist()
            elif type(sentence) == list:
                sentence = sentence
            else:
                raise RuntimeError("your input sen is wrong, it must be type of list or np.array")
            return self.model.predict(sentence)
        else:
            if type(sentence) == np.ndarray:
                sentence = sentence.tolist()
            elif type(sentence) == list:
                sentence = np.array([sentence])
            else:
                raise RuntimeError("your input sen is wrong, it must be type of list or np.array")
            return self.model.predict(sentence)

