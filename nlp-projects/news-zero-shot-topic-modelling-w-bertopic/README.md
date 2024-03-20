## News Zero-Shot Topic Modelling w/BERTopic

(kaggle link -> https://www.kaggle.com/code/banddaniel/news-zero-shot-topic-modelling-w-bertopic)

<i><span style="color:#e74c3c;"><b>MAIN GOAL: </b> I tried BERTopic model [1,2] for zero-shot topic modelling. I used 10 topics for modelling (5 categories in dataset and 5 for others). </span></i>


* I applied several <b>preprocessing</b> operations (cleaning, dropping stop words, lemmatization),
* I used <b>a DeBERTa based model </b> for the zero-hot classification [3],

<i>TODO: calculate topics coherence for evaluate the model.</i>

## Results

<img width="1070" alt="download (1)" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/da463375-4503-4530-8975-faa3dd171d18">

<br>
<img width="1067" alt="download (2)" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/07f63df3-ee91-4cd5-8a90-33acc24ae37d">



## My Another Projects
* [Complaint Analysis w/Ensemble Model (CatBoost, LR)](https://www.kaggle.com/code/banddaniel/complaint-analysis-w-ensemble-model-catboost-lr)
* [Gemma 2B Text Summarization w/Zero-Shot Prompting](https://www.kaggle.com/code/banddaniel/gemma-2b-text-summarization-w-zero-shot-prompting)
* [Towards Data Science Articles Topic Modeling w/LDA](https://www.kaggle.com/code/banddaniel/towards-data-science-articles-topic-modeling-w-lda)


## References
1. Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2203.05794
2. https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html
3. https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
