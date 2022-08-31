# -*- coding: utf-8 -*-
import io, os, sys
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM, Dropout
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras.callbacks import LambdaCallback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Bidirectional, TimeDistributed



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
    NER_TRAIN_PATH = sys.argv[1]
    NER_TEST_PATH = sys.argv[2]
    VEC_FILE_PATH = sys.argv[3]

    fin = io.open(VEC_FILE_PATH, 'r', encoding='utf-8', newline='\n', errors='ignore')

    print("loading...")

    word_vecs = {}

    for i, line in enumerate(fin):
         tokens = line.rstrip().split(' ')
         array = np.array(list(map(float, tokens[1:])))
         array = array / np.sqrt(np.sum(array * array + 1e-8))
         word_vecs[tokens[0]] = array

    # train_data = pd.read_csv(NER_TRAIN_PATH, header=0)
    # test_data = pd.read_csv(NER_TEST_PATH, header=0)

    ## 문장들만 모아둔 파일
    train_data = open(NER_TRAIN_PATH, encoding='utf-8')
    test_data = open(NER_TEST_PATH, encoding='utf-8')



    ## 문장 단위 단어리스트
    words = []

    ## 문장 단위 레이블리스트
    labels = []

    ## 전체 토큰 포함 이중리스트 (기존 리스트)
    data_list = []

    line = 1
    while (line):
        try:
            line = train_data.readline()
            sen = line
            sen = sen.replace('>',' ').replace('<',' ').replace('.','').replace('\n','').replace('\'','').split(' ')
            while '' in sen:
                sen.remove('')
            # print(sen)
            li = []
            for word in sen:
                tmp = []
                if word.find(':') > 0:
                    tmp = word.split(':')
                else:
                    tmp.append(word)
                    tmp.append('0')
                if len(tmp) != 2:
                    str = ":".join(tmp[:-1])
                    tmp = [str, tmp[-1]]
                    # print(tmp)
                li.append(tmp)

            data_list += li
            data_list += '*'
        except:
            continue

    tmp_words = []
    tmp_labels = []
    for l in data_list:
        if l == '*':
            if tmp_words != [] and tmp_labels != []:
                words.append(tmp_words)
                labels.append(tmp_labels)
            tmp_words = []
            tmp_labels = []
            continue
        else:
            tmp_words.append(l[0])
            tmp_labels.append(l[1])

    # 문장 개수 확인용
    print(len(words))
    print(len(labels))

    train_data.close()

    ## 문장 단위 단어리스트
    words2 = []

    ## 문장 단위 레이블리스트
    labels2 = []

    ## 전체 토큰 포함 이중리스트 (기존 리스트)
    data_list2 = []

    line = 1
    while (line):
        try:
            line = test_data.readline()
            sen = line
            sen = sen.replace('>',' ').replace('<',' ').replace('.','').replace('\n','').replace('\'','').split(' ')
            while '' in sen:
                sen.remove('')
            # print(sen)
            li = []
            for word in sen:
                tmp = []
                if word.find(':') > 0:
                    tmp = word.split(':')
                else:
                    tmp.append(word)
                    tmp.append('0')
                if len(tmp) != 2:
                    str = ":".join(tmp[:-1])
                    tmp = [str, tmp[-1]]
                    # print(tmp)
                li.append(tmp)

            data_list2 += li
            data_list2 += '*'
        except:
            continue

    tmp_words = []
    tmp_labels = []
    for l in data_list2:
        if l == '*':
            if tmp_words != [] and tmp_labels != []:
                words2.append(tmp_words)
                labels2.append(tmp_labels)
            tmp_words = []
            tmp_labels = []
            continue
        else:
            tmp_words.append(l[0])
            tmp_labels.append(l[1])

    # 문장 개수 확인용
    print(len(words2))
    print(len(labels2))

    test_data.close()

    result = []
    for x in labels:
        for tmp in x:
            if tmp not in result:
                result.append(tmp)

    result2 = []
    for x in labels2:
        for tmp in x:
            if tmp not in result2:
                result2.append(tmp)

    print(result)
    print(result2)

    print("Tokenizing...")

    src_tokenizer = Tokenizer(oov_token="<UNK>")
    src_tokenizer.fit_on_texts(words)
    x_train = src_tokenizer.texts_to_sequences(words)
    MAX_SEQUENCE_LENGTH = 20
    x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)

    tar_tokenizer = Tokenizer()
    tar_tokenizer.fit_on_texts(labels)

    tag_size = len(tar_tokenizer.word_index) + 1

    y_train = tar_tokenizer.texts_to_sequences(labels)
    y_train = pad_sequences(y_train, maxlen=MAX_SEQUENCE_LENGTH)
    y_train = to_categorical(y_train, num_classes=tag_size)

    # -----------------------------------------------------------------

    src2_tokenizer = Tokenizer(oov_token="<UNK>")
    src2_tokenizer.fit_on_texts(words2)
    x_test = src2_tokenizer.texts_to_sequences(words2)
    MAX_SEQUENCE_LENGTH = 20
    x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

    tar2_tokenizer = Tokenizer()
    tar2_tokenizer.fit_on_texts(labels2)

    tag_size2 = len(tar2_tokenizer.word_index) + 1

    y_test = tar2_tokenizer.texts_to_sequences(labels2)
    y_test = pad_sequences(y_test, maxlen=MAX_SEQUENCE_LENGTH)
    y_test = to_categorical(y_test, num_classes=tag_size2)

    word_size = len(src_tokenizer.word_index) + 1  # 1을 더해주는 것은 padding으로 채운 0 때문입니다
    EMBEDDING_DIM = 300

    embedding_matrix = np.zeros((word_size, EMBEDDING_DIM))

    # tokenizer에 있는 단어 사전을 순회하면서 word2vec의 300차원 vector를 가져옵니다
    for word, idx in src_tokenizer.word_index.items():
        embedding_vector = word_vecs[word] if word in word_vecs else None
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    model = Sequential()
    model.add(Embedding(word_size, 300, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], trainable=False, mask_zero=True))
    model.add(Bidirectional(LSTM(300, dropout=0.5, return_sequences=True)))
    model.add(TimeDistributed(Dense(tag_size, activation='softmax')))

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', precision, recall, f1score])

    model.fit(x_train, y_train, epochs=3)

    _loss, _acc, _precision, _recall, _f1score = model.evaluate(x_test, y_test)
    print('loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}'.format(_loss, _acc,
                                                                                                      _precision,
                                                                                                      _recall,
                                                                                                      _f1score))


if __name__ == "__main__":
    main()
