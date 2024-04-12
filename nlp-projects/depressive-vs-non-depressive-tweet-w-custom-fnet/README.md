## Depressive vs Non-depressive Tweet w/Custom FNet

(kaggle link -> https://www.kaggle.com/code/banddaniel/depressive-vs-non-depressive-tweet-w-custom-fnet)


<i><span style="color:#e74c3c;"><b>MAIN GOAL: </b> I tried to classify a depressive or a non-depressive tweet using a FNet (Mixing Tokens with Fourier Transforms) [1] model.</span></i>


* I applied several <b>preprocessing</b> operations (cleaning,dropping stop words etc.),
* Used <b>tf.data pipeline</b> for efficient training,
* I created <b>Vocabulary</b> and trained <b>Tokenizer</b> using the train data,
* 5 KFold <b>cross validation</b>,
* I have modified this notebook [2],

## Proposed Model
<img width="487" alt="Screenshot 2024-04-13 at 12 30 05 AM" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/3042d1e7-2d33-41c2-ad55-eeac4b3206da">

## Test Results
![__results___23_1](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/95c52e09-729d-454a-8641-539ebbcd7860)


## References
1. Lee-Thorp, J., Ainslie, J., Eckstein, I., & Ontanon, S. (2021). FNet: Mixing Tokens with Fourier Transforms (Version 4). arXiv. https://doi.org/10.48550/ARXIV.2105.03824
2. https://keras.io/examples/nlp/fnet_classification_with_keras_nlp/
