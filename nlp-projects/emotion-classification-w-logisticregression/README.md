## Emotion Classification w/LogisticRegression

(kaggle link -> https://www.kaggle.com/code/banddaniel/emotion-classification-w-logisticregression)

*I tried to predict an emotion with LogisticRegression.*
​
* Dropped duplicate samples (original data size <b>839555</b>, after dropped <b>393822</b>),
* I applied several <b>preprocessing</b> operations (cleaning,dropping stop words),
* I tried a hybrid model <b>(SentenceTransformer + PCA + CatBoostClassifier)</b>, the model produce nearly 0.9 F1 score. 
​
​
<i> <b>Result: </b> <span style="color:#e74c3c;">I checked target leakage for an overfitting issue by calculating cosine similarities between train samples and test samples, but there was no considerable leakage (generally, at most cosine similarity is nearly 80-90% for only 4-5 samples). </span> </i>
​

### Results
![__results___14_1](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/5c1e07b2-e46c-42c0-b2f9-170252a133a7)
