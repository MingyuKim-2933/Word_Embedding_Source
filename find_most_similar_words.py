import io
import sys
import numpy as np


def cos_similarity(x, y, eps=1e-8):
    '''코사인 유사도 산출
    :param x: 벡터
    :param y: 벡터
    :param eps: '0으로 나누기'를 방지하기 위한 작은 값
    :return:
    '''

    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_vec, top=5):
    '''유사 단어 검색
    :param query: 쿼리(텍스트)
    :param word_to_vec: word_tovec
    :param top: 상위 몇 개까지 출력할 지 지정
    '''
    if query not in word_to_vec:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('\n[query] ' + query)

    word_to_vec_keys = list(word_to_vec.keys())

    # 코사인 유사도 계산
    vocab_size = len(word_to_vec)

    query_vec = word_to_vec[query]
    similarity = np.zeros(vocab_size)
    cnt = 0
    for i in word_to_vec.values():
        similarity[cnt] = cos_similarity(i, query_vec)
        cnt += 1

    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if word_to_vec_keys[i] == query:
            continue
        print(' %s: %s' % (word_to_vec_keys[i], similarity[i]))

        count += 1
        if count >= top:
            return


def main():
    vec_file_path = sys.argv[1]
    word = sys.argv[2]
    fin = io.open(vec_file_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vecs = {}

    for i, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        array = np.array(list(map(float, tokens[1:])))
        array = array / np.sqrt(np.sum(array * array + 1e-8))
        word_vecs[tokens[0]] = array

    most_similar(word, word_vecs, 5)


if __name__ == "__main__":
    main()
