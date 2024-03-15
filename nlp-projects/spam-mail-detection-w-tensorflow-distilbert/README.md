## Spam Mail Detection w/Tensorflow (DistilBERT)

*I tried to predict a spam mail with finetuning a DistilBert based Tensorflow model.*

* I applied several <b>preprocessing</b> operations (cleaning,dropping stop words),
* Used tf.data pipeline for efficient training,
* I only used only 20 max length for sequence length (bert models support up to 512 input lengths),
* Only 18000 samples be used for training (12000 samples for validating and 20000 samples for testing),


<img width="1134" alt="Screenshot 2024-03-14 at 8 45 08 PM" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/d169e959-c215-4dd5-a217-e1a78201aedb">



## My Another Projects
* [Complaint Analysis w/Ensemble Model (CatBoost, LR)](https://www.kaggle.com/code/banddaniel/complaint-analysis-w-ensemble-model-catboost-lr)
* [Gemma 2B Text Summarization w/Zero-Shot Prompting](https://www.kaggle.com/code/banddaniel/gemma-2b-text-summarization-w-zero-shot-prompting)
* [Rating Prediction w/SentenceTransformer, CatBoost](https://www.kaggle.com/code/banddaniel/rating-prediction-w-sentencetransformer-catboost)


## References
1. https://towardsdatascience.com/hugging-face-transformers-fine-tuning-distilbert-for-binary-classification-tasks-490f1d192379
2. https://www.kaggle.com/code/preatcher/emotion-detection-by-using-bert
