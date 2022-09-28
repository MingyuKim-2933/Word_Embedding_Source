# -*- coding: utf-8 -*-
import io, os, sys
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM, Dropout
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from datasets import load_dataset


def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())

    # return a single tensor value
    return _f1score


def main():
    YNAT_TRAIN_PATH = sys.argv[1]
    YNAT_TEST_PATH = sys.argv[2]
    VEC_FILE_PATH = sys.argv[3]

    task = "ynat"

    dataset = load_dataset('klue', task)

    fin = io.open(VEC_FILE_PATH, 'r', encoding='utf-8', newline='\n', errors='ignore')

    print("loading...")

    word_vecs = {}

    for i, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        array = np.array(list(map(float, tokens[1:])))
        array = array / np.sqrt(np.sum(array * array + 1e-8))
        word_vecs[tokens[0]] = array

    x_train = []
    train_text = open(YNAT_TRAIN_PATH, mode='r')
    line = None
    while line != ' ':
        line = train_text.readline()
        x_train.append(line)
    train_text.close()

    y_train = dataset['train']['label']

    x_test = []
    test_text = open(YNAT_TEST_PATH, mode='r')
    line = None
    while line != ' ':
        line = test_text.readline()
        x_test.append(line)
    test_text.close()

    y_test = dataset['validation']['label']

    print("Tokenizing...")

    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(x_train)
    train_sequence = tokenizer.texts_to_sequences(x_train)
    MAX_SEQUENCE_LENGTH = 20
    train_inputs = pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    train_labels = np.array(y_train)

    test_sequence = tokenizer.texts_to_sequences(x_test)
    test_inputs = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    test_labels = np.array(y_test)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    word_size = len(tokenizer.word_index) + 1  # 1을 더해주는 것은 padding으로 채운 0 때문입니다
    EMBEDDING_DIM = 300

    embedding_matrix = np.zeros((word_size, EMBEDDING_DIM))

    # tokenizer에 있는 단어 사전을 순회하면서 word2vec의 300차원 vector를 가져옵니다
    for word, idx in tokenizer.word_index.items():
        embedding_vector = word_vecs[word] if word in word_vecs else None
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    num_classes = 7

    model = Sequential()
    model.add(Embedding(word_size, 300, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], trainable=False))
    model.add(LSTM(300, dropout=0.5))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', precision, recall, f1score])

    model.fit(train_inputs, train_labels, epochs=10)

    _loss, _acc, _precision, _recall, _f1score = model.evaluate(test_inputs, test_labels)
    print('loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}'.format(_loss, _acc,
                                                                                                      _precision,
                                                                                                      _recall,
                                                                                                      _f1score))

if __name__ == "__main__":
    main()
