#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 11:28
# @Author  : Leslee
import os
import json
import jieba
import codecs
import numpy as np
import keras.backend as K
from keras.models import Input, Model
from gensim.models import KeyedVectors
from keras.layers import Add, Embedding

from src.keras_layers.non_mask_layer import NonMaskingLayer
from src.data_utils.TextProcUtil import get_ngram, extract_chinese
from src.conf.path_config import path_embedding_random_char, path_embedding_random_word, path_embedding_bert, \
    path_embedding_xlnet, path_embedding_albert, path_embedding_vector_word2vec_char, \
    path_embedding_vector_word2vec_word


class BaseEmbedding:
    def __init__(self, hyper_parameters):
        self.len_max = hyper_parameters.get('len_max', 50)
        self.embed_size = hyper_parameters.get('embed_size', 300)
        self.vocab_size = hyper_parameters.get('vocab_size', 30000)
        self.trainable = hyper_parameters.get('trainable', False)
        self.level_type = hyper_parameters.get('level_type', 'char')
        # 词嵌入方式，可以选择'xlnet'、'bert'、'random'、'word2vec'
        self.embedding_type = hyper_parameters.get('embedding_type', 'word2vec')

        if self.level_type == 'word':
            if self.embedding_type == "random":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_random_word)
            elif self.embedding_type == "word2vec":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_vector_word2vec_word)
            elif self.embedding_type == 'bert':
                raise RuntimeError("bert level_type is 'char', not 'word'")
            elif self.embedding_type == "xlnet":
                raise RuntimeError("xlnet level_type is 'char', not 'word'")
            elif self.embedding_type == "albert":
                raise RuntimeError("albert level_type is 'char', not 'word'")
            else:
                raise RuntimeError("embedding_type must be 'random', 'word2vec' or 'bert'")
        elif self.level_type == "char":
            if self.embedding_type == "random":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_random_char)
            elif self.embedding_type == "word2vec":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_vector_word2vec_char)
            elif self.embedding_type == "bert":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_bert)
            elif self.embedding_type == "xlnet":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_xlnet)
            elif self.embedding_type == "albert":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_albert)
            else:
                raise RuntimeError("embedding_type must be 'random', 'word2vec' or 'bert'")
        elif self.level_type == "ngram":
            if self.embedding_type == 'random':
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path')
                if not self.corpus_path:
                    raise RuntimeError("corpus_path does not Exists!")
            else:
                raise RuntimeError("embedding_type must be 'random', 'word2vec' or 'bert'")
        else:
            raise RuntimeError("level_type must be 'char' or 'word'")
        # 模型需要的符号
        self.ot_dict = {'[PAD]': 0,
                        '[UNK]': 1,
                        '[BOS]': 2,
                        '[EOS]': 3, }
        self.deal_corpus()
        self.build()

    def deal_corpus(self):
        pass

    def build(self):
        self.token2idx = {}
        self.idx2token = {}

    def sentence2idx(self, text, second_text=None):
        if second_text:  # bert中的second sentence
            second_text = "[SEP]" + str(second_text).upper()
        text = str(text).upper()

        if self.level_type == "char":
            text = list(text)
        elif self.level_type == "word":
            text = list(jieba.cut(text, cut_all=False, HMM=True))
        else:
            raise RuntimeError("input level_type is wrong, it must be 'word' or 'char'")
        text = [text_one for text_one in text]
        len_leave = self.len_max - len(text)
        if len_leave >= 0:
            # 如果句子不够最大长度，用PAD进行补全
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]']
                          for text_char in text] + [self.token2idx['[PAD]'] for i in range(len_leave)]
        else:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text[0:self.len_max]]
        return text_index

    def idx2sentence(self, idx):
        assert type(idx) == list
        text_idx = [self.idx2token[id] if id in self.idx2token else self.idx2token['[UNK]'] for id in idx]
        return "".join(text_idx)


class RandomEmbedding(BaseEmbedding):

    def __init__(self, hyper_parameters):
        self.ngram_ns = hyper_parameters['embedding'].get('ngram_ns', [1, 2, 3])  # ngram信息
        super().__init__(hyper_parameters=hyper_parameters)

    def deal_corpus(self):
        token2idx = self.ot_dict.copy()
        count = 3
        if 'term' in self.corpus_path:
            with open(file=self.corpus_path, mode='r', encoding='utf-8') as fd:
                while True:
                    term_one = fd.readline()
                    if not term_one:
                        break
                    term_one = term_one.strip()
                    if term_one not in token2idx:
                        count = count + 1
                        token2idx[term_one] = count
        elif os.path.exists(self.corpus_path):
            with open(file=self.corpus_path, mode='r', encoding='utf-8') as fd:
                terms = fd.readlines()
                for term_one in terms:
                    if self.level_type == 'char':
                        text = list(term_one.replace(' ', '').strip())
                    elif self.level_type == 'word':
                        text = list(jieba.cut(term_one, cut_all=False, HMM=False))
                    elif self.level_type == "ngram":
                        text = get_ngram(term_one, ns=self.ngram_ns)
                    else:
                        raise RuntimeError("your input level_type is wrong, it must be 'word', 'char', 'ngram'")
                    for text_one in text:
                        if text_one not in token2idx:
                            count = count + 1
                            token2idx[text_one] = count
        else:
            raise RuntimeError("input corpus_path is wrong, it must be 'dict' or 'corpus'")
        self.token2idx = token2idx
        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.idx2token[value] = key

    # 构建embedding层model
    def build(self, **kwargs):
        self.vocab_size = len(self.token2idx)
        self.input = Input(shape=(self.len_max,), dtype='int32')
        self.output = Embedding(self.vocab_size+1,
                                self.embed_size,
                                input_length=self.len_max,
                                trainable=self.trainable)
        self.model = Model(self.input, self.output)

    def sentence2idx(self, text, second_text=""):
        if second_text:
            second_text = "[SEP]" + str(second_text).upper()
        text = str(text).upper() + second_text
        if self.level_type == 'char':
            text = list(text)
        elif self.level_type == 'word':
            text = list(jieba.cut(text, cut_all=False, HMM=False))
        elif self.level_type == 'ngram':
            text = get_ngram(text, ns=self.ngram_ns)

class WordEmbedding(BaseEmbedding):

    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters)

    def build(self, **kwargs):
        self.embedding_type = 'word2vec'
        print("load word2vec start")
        self.key_vectors = KeyedVectors.load_word2vec_format(self.corpus_path, **kwargs)
        print("loaded word2vec Model!")
        self.embed_size = self.key_vectors.vector_size

        self.token2idx = self.ot_dict.copy()
        embedding_matrix = []
        embedding_matrix.append(np.zeros(self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))

        for word in self.key_vectors.index2entity:
            self.token2idx[word] = len(self.token2idx)  # 确认一下
            embedding_matrix.append(self.key_vectors[word])

        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.token2idx[word] = len(self.token2idx)

        self.vocab_size = len(self.token2idx)
        embedding_matrix = np.array(embedding_matrix)
        self.input = Embedding(self.vocab_size, self.embed_size,
                               input_length=self.len_max, weights=[embedding_matrix],
                               trainable=self.trainable)(self.input)
        self.output = Embedding(self.vocab_size, self.embed_size,
                                input_length=self.len_max, weights=[embedding_matrix],
                                trainable=self.trainable)(self.input)
        self.model = Model(self.input, self.output)


class BertEmbedding(BaseEmbedding):

    def __init__(self, hyper_parameters):
        self.layer_indexes = hyper_parameters['embedding'].get('layer_indexes', [12])
        super().__init__(hyper_parameters)

    def build(self):
        import keras_bert
        self.embedding_type = 'bert'
        config_path = os.path.join(self.corpus_path, 'bert_config.json')
        check_point_path = os.path.join(self.corpus_path, 'bert_model.ckpt')
        dict_path = os.path.join(self.corpus_path, 'vocab.txt')
        print("Load Bert Model Start!")
        model = keras_bert.load_trained_model_from_checkpoint(config_path, check_point_path,
                                                              seq_len=self.len_max, trainable=self.trainable)
        print("loaded bert Model!")
        #  取出bert的所有的层
        layer_dict = [6]
        layer_0 = 7
        for i in range(12):
            layer_0 = layer_0 + 8
            layer_dict.append(layer_0)
        print(layer_dict)
        if len(self.layer_indexes) == 0:
            encoder_layer = model.output
        # 分类如果只有一层，就只取最后一层的weight；
        elif len(self.layer_indexes) == 1:
            if self.layer_indexes[0] in [i + 1 for i in range(13)]:
                encoder_layer = model.get_layer(index=layer_dict[self.layer_indexes[0] - 1]).output
            else:
                encoder_layer = model.get_layer(index=layer_dict[-1]).output
        # 遍历需要的层， 所有层的weight取出来，拼接成：768*层数
        else:
            all_layers = [model.get_layer(index=layer_dict[lay - 1]).output if lay in [i + 1 for i in range(13)]
                          else model.get_layer(index=layer_dict[-1]).output for lay in self.layer_indexes]
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = Add()(all_layers_select)
        self.output = NonMaskingLayer()(encoder_layer)
        self.input = model.inputs
        self.model = Model(self.input, self.output)
        self.embedding_size = self.model.output_shape[-1]

        self.token_dict = {}
        with codecs.open(dict_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        self.vocab_size = len(self.token_dict)
        self.tokenizer = keras_bert.Tokenizer(self.token_dict)

    # todo： 加载bert4kereas
    def build_keras4bert(self):
        pass

    def sentence2idx(self, text, second_text=None):
        text = extract_chinese(str(text.upper()))
        text = str(text).upper()
        input_id, input_type_id = self.tokenizer.encode(first=text, second=second_text, max_len=self.len_max)
        return [input_id, input_type_id]


class XlnetEmbedding(BaseEmbedding):

    def __init__(self, hyper_parameters):
        pass

    def build_config(self):
        pass

    def build(self):
        pass


class AlbertEmbedding(BaseEmbedding):

    def __init__(self, hyper_parameters):
        self.layer_indexes = hyper_parameters['embedding'].get('layer_indexes', [12])
        super().__init__(hyper_parameters)

    def build(self):
        from src.keras_layers.albert.albert import load_brightmart_albert_zh_checkpoint
        import keras_bert
        self.embedding_type = 'albert'
        dict_path = os.path.join(self.corpus_path, 'vocab.txt')
        print("Load Albert Model Start!")
        self.model = load_brightmart_albert_zh_checkpoint(self.corpus_path, training=self.trainable,
                                                          seq_len=self.len_max, output_layers=None)
        config = {}
        for file_name in os.listdir(self.corpus_path):
            if file_name.startswith('bert_config.json'):
                with open(os.path.join(self.corpus_path, file_name)) as reader:
                    config = json.load(reader)
                break

        num_hidden_layers = config.get('num_hidden_layers', 0)
        layer_real = [i for i in range(num_hidden_layers)] + [-i for i in range(num_hidden_layers)]
        self.layer_indexes = [i if i in layer_real else -2 for i in self.layer_indexes]
        model_1 = self.model.layers
        print('load bert model end!')
        # 取出所有的albert的层
        layer_dict = [4, 8, 11, 13]
        layer_0 = 13
        for i in range(num_hidden_layers):
            layer_0 = layer_0 + 1
            layer_dict.append(layer_0)
        print(layer_dict)
        if len(self.layer_indexes) == 0:
            encoder_layer = self.model.output
        # 取最后一层权重，默认最后一层
        elif len(self.layer_indexes) == 1:
            all_layers = [self.model.get_layer(index=layer_dict[lay]).output if lay in layer_real
                          else self.model.get_layer(index=layer_dict[-2]).output
                          for lay in self.layer_indexes]
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = Add()(all_layers_select)
        self.output = NonMaskingLayer()(encoder_layer)
        self.input = self.model.inputs
        self.model = Model(self.input, self.output)

        self.token_dict = {}
        with codecs.open(dict_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        self.vocab_size = len(self.token_dict)
        self.tokenizer = keras_bert.Tokenizer(self.token_dict)

    def sentence2idx(self, text, second_text=''):
        text = str(text).upper()
        input_id, input_type_id = self.tokenizer.encode(first=text, second=second_text, max_len=self.len_max)
        return [input_id, input_type_id]
