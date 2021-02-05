#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/2/4 16:25
# @Author  : Leslee
from keras.layers import Dense, Dropout, Flatten, Input, Bidirectional, GRU
from keras.models import Model
from keras import regularizers
from keras.models import Model

from src.model.BaseGraph import Graph
from src.keras_layers.attention_self import AttentionSelf


class HANModel(Graph):

    def __init__(self, hyper_parameters):
        self.rnn_type = hyper_parameters['model'].get('rnn_type', 'Bidirectional-LSTM')
        self.rnn_units = hyper_parameters['model'].get('rnn_units', 256)
        self.attention_units = hyper_parameters['model'].get('attention_units', self.rnn_units*2)
        self.dropout_spatial = hyper_parameters['model'].get('droupout_spatial', 0.2)
        self.len_max_sen = hyper_parameters['model'].get('len_max_sen', 50)
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):

        super().create_model(hyper_parameters)
        x_input_word = self.word_embedding.output
        x_word = self.word_level()(x_input_word)
        x_word_to_sent = Dropout(self.dropout)(x_word)

        # 句子或文档level
        x_sentence = self.sententce_level()(x_word_to_sent)
        x_sentence = Dropout(self.dropout)(x_sentence)
        x_sent = Flatten()(x_sentence)
        dense_layer = Dense(self.label, activation=self.activate_classify)(x_sentence)
        output = [dense_layer]
        self.model = Model(self.word_embedding.input, output)
        self.model.summary(128)

    def word_level(self):
        x_input_word = Input(shape=(self.len_max, self.embed_size))
        x = Bidirectional(GRU(units=self.rnn_units,
                              return_sequences=True,
                              activation='relu',
                              kernel_regularizer=regularizers.l2(self.l2),
                              recurrent_regularizer=regularizers.l2(self.l2)))(x_input_word)
        out_sent = AttentionSelf(self.rnn_units*2)(x)
        model = Model(x_input_word, out_sent)
        return model

    def sententce_level(self):
        x_input_sen = Input(shape=(self.len_max, self.rnn_units*2))
        output_doc = Bidirectional(GRU(units=self.rnn_units*2,
                                       return_sequences=True,
                                       activation='relu',
                                       kernel_regularizer=regularizers.l2(self.l2),
                                       recurrent_regularizer=regularizers.l2(self.l2)))(x_input_sen)
        output_doc_att = AttentionSelf(self.word_embedding.embed_size)(output_doc)
        model = Model(x_input_sen, output_doc_att)
        return model
