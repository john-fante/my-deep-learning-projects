## Complaint Analysis w/Ensemble Model (CatBoost, LR)

(kaggle link -> https://www.kaggle.com/code/banddaniel/complaint-analysis-w-ensemble-model-catboost-lr)

*I tried to predict the complaint classification of bank customers with an ensemble-based model.*

First of all, I applied several <b>preprocessing</b> operations (I tried stemming and lemmatization, but there was no betterment in the F1 score), then following models and methods

<hr>

#### <span style="color:#e74c3c;"> Model 1 </span> Transformer Features, CatBoostClassifier
* I used <b>a pretrained embeddings</b> for the text feature extraction stage [1],
* <b>PCA for dimensionality reduction </b> (applied the output of pretrained embeddings with 300 components),

#### <span style="color:#e74c3c;"> Model 2 </span> TfidfVectorizer, Logistic Regression
* Created text features with <b>TfidfVectorizer</b>
* Predicted with Logistic Regression (with simple tuned parameter)

#### <span style="color:#2980b9;"> Final Weighted Average Ensemble Model</span> 
* Used the first model weight of 32 % and the second model weight of 68 %

<hr>

<img width="1255" alt="Screenshot 2024-03-03 at 10 31 57 PM" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/f80a7fb0-c091-473b-87cf-e6f817a27cb9">

*Figure 1: proposed ensemble classification pipeline*


## My Another Projects
* [Gemma 2B Text Summarization w/Zero-Shot Prompting](https://www.kaggle.com/code/banddaniel/gemma-2b-text-summarization-w-zero-shot-prompting)
* [Rating Prediction w/SentenceTransformer, CatBoost](https://www.kaggle.com/code/banddaniel/rating-prediction-w-sentencetransformer-catboost)
* [Sentiment Analysis w/CatBoostClassifier](https://www.kaggle.com/code/banddaniel/sentiment-analysis-w-catboostclassifier)

## References
1. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
