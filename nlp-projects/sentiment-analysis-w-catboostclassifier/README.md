## Sentiment Analysis w/CatBoostClassifier

(kaggle link -> https://www.kaggle.com/code/banddaniel/sentiment-analysis-w-catboostclassifier)

*I tried to predict emotion classification with CatBoostClassifier.*

* I applied several <b>preprocessing</b> operations (I tried stemming and lemmatization, but there was no betterment in the F1 score),
* I used <b>a pretrained embeddings</b> for the text feature extraction stage [1],
* <b>PCA for dimensionality reduction </b> (applied the output of pretrained embeddings with 300 components),
* Used a tuned <b> CatBoostClassifier</b> for rating predictions (tuned with optuna)

<img width="1257" alt="Screenshot 2024-03-02 at 4 18 03 PM" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/ebeb4b1b-ed60-47a4-b6c2-5b35af19c469">
*Figure 1: proposed classification pipeline*

## References
1. https://huggingface.co/sentence-transformers/all-mpnet-base-v2
