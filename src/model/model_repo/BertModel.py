#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/2/4 15:04
# @Author  : Leslee
from __future__ import print_function, division
from keras.models import Model
from keras.layers import Flatten, Dense, Lambda

from src.model.BaseGraph import Graph


class BertModel(Graph):

    def __init__(self, hyper_parameters):

        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):

        super().create_model(hyper_parameters)
        embedding_output = self.word_embedding.output
        x = Lambda(lambda x: x[:, 0:1, :])(embedding_output)
        x = Flatten()(x)
        dense_layer = Dense(self.label, activation=self.activate_classify)(x)
        output_layers = [dense_layer]
        self.model = Model(self.word_embedding.input, output_layers)
        self.model.summary(120)
