## Disease Article Topic Modelling w/BERTopic

(kaggle link -> https://www.kaggle.com/code/banddaniel/disease-article-topic-modelling-w-bertopic)


<i><span style="color:#e74c3c;"><b>MAIN GOAL: </b> I tried BERTopic model [1,2] for zero-shot topic modelling. I used <b>35 topics</b> for modelling.</span></i>


* I applied several <b>preprocessing</b> operations (cleaning, dropping stop words, lemmatization),
* A function for creating candidate topics (unique Disease Name),
* I used <b>a DeBERTa based model </b> for the zero-hot classification [3],


<i>TODO: calculate topics coherence for evaluate the model.</i>


## Results


<img width="788" alt="Screenshot 2024-03-25 at 2 20 04 PM" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/75a0906a-b55f-40f8-b424-3f0435596949">

<br>
<img width="789" alt="Screenshot 2024-03-25 at 2 19 23 PM" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/803d1449-ce84-4dfa-8a12-7fc66fff8f2f">


## References
1. Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2203.05794
2. https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html
3. https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
