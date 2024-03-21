## Spam vs Ham Message Classification w/Custom FNet

(kaggle link -> https://www.kaggle.com/code/banddaniel/spam-vs-ham-message-classification-w-custom-fnet)

<i><span style="color:#e74c3c;"><b>MAIN GOAL: </b>  I tried to classify spam or ham message FNet (Mixing Tokens with Fourier Transforms) [1] model.</span></i>


* I applied several <b>preprocessing</b> operations (cleaning,dropping stop words etc.),
* Used <b>tf.data pipeline</b> for efficient training,
* I created <b>Vocabulary</b> and trained <b>Tokenizer</b> using the train data,
* I have modified this notebook [2],

## Proposed Model
![output-onlinepngtools-2](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/aa510807-87f0-4087-86da-b409bd74ac1e)



## My Another Projects
* [News Zero-Shot Topic Modelling w/BERTopic](https://www.kaggle.com/code/banddaniel/news-zero-shot-topic-modelling-w-bertopic)
* [Complaint Analysis w/Ensemble Model (CatBoost, LR)](https://www.kaggle.com/code/banddaniel/complaint-analysis-w-ensemble-model-catboost-lr)
* [Gemma 2B Text Summarization w/Zero-Shot Prompting](https://www.kaggle.com/code/banddaniel/gemma-2b-text-summarization-w-zero-shot-prompting)



## References
1. Lee-Thorp, J., Ainslie, J., Eckstein, I., & Ontanon, S. (2021). FNet: Mixing Tokens with Fourier Transforms (Version 4). arXiv. https://doi.org/10.48550/ARXIV.2105.03824
2. https://keras.io/examples/nlp/fnet_classification_with_keras_nlp/

