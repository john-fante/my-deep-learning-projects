## Rating Prediction w/SentenceTransformer, CatBoost

(kaggle link -> https://www.kaggle.com/code/banddaniel/rating-prediction-w-sentencetransformer-catboost)

**I tried to predict ratings with CatBoostRegressor.***

* Applied several preprocessing operations,
* I used a pretrained embeddings for the text feature extraction stage [1],
* Used a tuned CatBoostRegressor for rating predictions (tuned with optuna)


## References
1. https://huggingface.co/sentence-transformers/all-mpnet-base-v2
