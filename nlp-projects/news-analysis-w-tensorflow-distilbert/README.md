## News Analysis w/Tensorflow (TFDistilBERT)

(kaggle link -> https://www.kaggle.com/code/banddaniel/news-analysis-w-tensorflow-distilbert)

*I tried to predict a news category with a DistilBert based Tensorflow model.*

* I applied several <b>preprocessing</b> operations (cleaning,dropping stop words, lemmatizing),
* Used tf.data pipeline for efficient training,
* I only used 100 max length for sequence length (bert models support up to 512 input lengths)
* Only 1065 samples be used for training,  (710 samples for validating and 2664 samples for testing)



<img width="946" alt="Screenshot 2024-03-11 at 12 25 25 PM-min" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/46afdcc6-eae8-40d3-b4e1-6ba3fc63a277">


## References
1. https://towardsdatascience.com/hugging-face-transformers-fine-tuning-distilbert-for-binary-classification-tasks-490f1d192379
2. https://www.kaggle.com/code/preatcher/emotion-detection-by-using-bert
