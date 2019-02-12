# Core NLP Applications
- Search & Information Retrieval
- Document classification
- Spell checking, keyword search, finding synonyms, parsing
- Sentiment Analysis
- Question Answering
- Conversation Modeling
- Image captioning
- Machine Translation
- Summarization
- Speech recognition
- Entity recognition

# Characteristics of natural language
- Symbolic/categorical
  - data sparsity
  - but 사람이 언어처리할때는 뇌내의 continuous signal 사용
    - How to convert discrete symbols into continuous signals?

- Sequential
  - 심볼의 순서가 의미에 영향을 미침
- Compositional
  - 단어 + 단어 = 문구
  - 문구 + 문구 = 문장
  - 문장 + 문장 = 문단
  - 문단 + 문단 = 글
- Contextual
  - 앞뒤 맥락에 따라 같은 단어/문구도 의미가 달라짐

# 기타 팁
- Text classification in practice: bag-of-ngram words(e.g. FastTest) works the best. [조경현 교수님 connect 강의](https://www.edwith.org/deepnlp/lecture/29208/)의 3:30초 근처 볼것.
- CNN도 나쁘지 않지만, small-range local dependency만 보기 때문에 long-range dependency 모델링에는 한계가 있다. 특히 한글처럼 주어랑 동사 저 멀리 떨어져있고 한 경우에는 좋지 않을 수도 있음
- generalized CNN: Relation Network -> Self-attention
- 실제로는 RNN, CNN, subword model 등과 함께 섞어서 많이 쓴다.