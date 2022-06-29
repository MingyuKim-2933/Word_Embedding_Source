# -*- coding: utf-8 -*-
import io, os, sys
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM, Dropout
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
# from keras import backend as K
# from keras.callbacks import LambdaCallback



# def recall(y_target, y_pred):
#     # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
#     # round : 반올림한다
#     y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
#     y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
#
#     # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
#     count_true_positive = K.sum(y_target_yn * y_pred_yn)
#
#     # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
#     count_true_positive_false_negative = K.sum(y_target_yn)
#
#     # Recall =  (True Positive) / (True Positive + False Negative)
#     # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
#     recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
#
#     # return a single tensor value
#     return recall
#
#
# def precision(y_target, y_pred):
#     # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
#     # round : 반올림한다
#     y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
#     y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
#
#     # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
#     count_true_positive = K.sum(y_target_yn * y_pred_yn)
#
#     # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
#     count_true_positive_false_positive = K.sum(y_pred_yn)
#
#     # Precision = (True Positive) / (True Positive + False Positive)
#     # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
#     precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())
#
#     # return a single tensor value
#     return precision
#
#
# def f1score(y_target, y_pred):
#     _recall = recall(y_target, y_pred)
#     _precision = precision(y_target, y_pred)
#     # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
#     _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())
#
#     # return a single tensor value
#     return _f1score


def main():
    TRAIN_PATH = sys.argv[1]
    TEST_PATH = sys.argv[2]
    VEC_FILE_PATH = sys.argv[3]

    fin = io.open(VEC_FILE_PATH, 'r', encoding='utf-8', newline='\n', errors='ignore')

    print("loading...")

    word_vecs = {}

    for i, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        array = np.array(list(map(float, tokens[1:])))
        array = array / np.sqrt(np.sum(array * array + 1e-8))
        word_vecs[tokens[0]] = array

    train_data = pd.read_csv(TRAIN_PATH, names=['ratings', 'reviews', 'text'])
    test_data = pd.read_csv(TEST_PATH, names=['ratings', 'reviews', 'text', 1, 2, 3, 4])
    train_text = train_data['reviews']
    train_text = train_text.astype(str)
    train_data['label'] = train_data['ratings'] - 1
    train_label = np.array(train_data['label'])

    test_text = test_data['reviews']
    test_text = test_text.astype(str)
    test_data['label'] = test_data['ratings'] - 1
    test_label = np.array(test_data['label'])

    print("Tokenizing...")

    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(train_text)
    train_sequence = tokenizer.texts_to_sequences(train_text)
    MAX_SEQUENCE_LENGTH = 30
    train_inputs = pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    train_labels = np.array(train_label)

    test_sequence = tokenizer.texts_to_sequences(test_text)
    test_inputs = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    test_labels = np.array(test_label)

    word_size = len(tokenizer.word_index) + 1  # 1을 더해주는 것은 padding으로 채운 0 때문입니다
    EMBEDDING_DIM = 300

    embedding_matrix = np.zeros((word_size, EMBEDDING_DIM))

    # tokenizer에 있는 단어 사전을 순회하면서 word2vec의 300차원 vector를 가져옵니다
    for word, idx in tokenizer.word_index.items():
        embedding_vector = word_vecs[word] if word in word_vecs else None
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    model = Sequential()
    model.add(Embedding(word_size, 300, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], trainable=False))
    model.add(LSTM(300, dropout=0.5))
    model.add(Dense(5, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_inputs, train_labels, epochs=10)

    _loss, _acc = model.evaluate(test_inputs, test_labels)
    print('loss: {:.3f}, accuracy: {:.3f}'.format(_loss, _acc))


if __name__ == "__main__":
    main()
