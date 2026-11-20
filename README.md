# Word Embedding Source

미리 학습한 단어 임베딩(300차원, 텍스트 `.vec` 형식)의 품질을 여러 다운스트림 태스크로 평가하는 스크립트 모음입니다.

모든 스크립트는 같은 방식으로 동작합니다.
1. 임베딩 파일을 읽어 단어마다 L2 정규화된 벡터를 만듭니다.
2. 태스크 데이터로 단어 사전을 만들고, 사전의 각 단어에 해당 임베딩을 넣은 임베딩 행렬을 구성합니다. 임베딩에 없는 단어는 0 벡터로 둡니다.
3. 임베딩 층을 **고정(trainable=False)** 한 채로 LSTM 분류기를 학습합니다. 임베딩 자체의 품질만 비교하기 위해서입니다.

## 평가 태스크

| 스크립트 | 데이터 | 언어 | 문제 | 모델 | 최대 길이 |
|---|---|---|---|---|---|
| `KLUE-YNAT.py` | KLUE YNAT | 한국어 | 뉴스 주제 7분류 | LSTM(300) | 20 |
| `KLUE-NER.py` | KLUE NER | 한국어 | 개체명 인식 (토큰 단위) | BiLSTM(300) + TimeDistributed | 20 |
| `nsmc.py` | NSMC | 한국어 | 영화 리뷰 감성 이진 분류 | LSTM(300) | 30 |
| `ag_news_text_classification.py` | AG News | 영어 | 뉴스 4분류 | LSTM(300) | 100 |
| `amazon_f_text_classification.py` | Amazon Review Full | 영어 | 별점 5분류 | LSTM(300) | 30 |
| `amazon_p_text_classification.py` | Amazon Review Polarity | 영어 | 긍정, 부정 2분류 | LSTM(300) | 30 |

- 정확도와 함께 precision, recall, F1을 직접 정의한 Keras metric으로 출력합니다.

## 유사 단어 확인

`find_most_similar_words.py`는 단어 목록 파일을 받아 각 단어와 코사인 유사도가 가장 높은 단어 10개를 출력합니다. 자모 단위로 분해된 임베딩을 다시 글자로 조합해서 보여주기 때문에 `decompose` 모듈이 필요합니다. 이 모듈은 저장소에 포함되어 있지 않습니다.

## 실행 예시

```bash
# 한국어 뉴스 주제 분류 (형태소 분석된 문장 파일 + 임베딩)
python KLUE-YNAT.py <train_sentences.txt> <validation_sentences.txt> <embedding.vec>

# 영화 리뷰 감성 분류
python nsmc.py <ratings_train.txt> <ratings_test.txt> <parsed_train.txt> <parsed_test.txt> <embedding.vec>

# 유사 단어 확인
python find_most_similar_words.py <embedding.vec> <words.txt>
```

- KLUE 레이블은 Hugging Face `datasets`의 `load_dataset('klue', ...)`로 불러옵니다.

## 기술 스택

Python, TensorFlow/Keras, Hugging Face datasets, NumPy, pandas
