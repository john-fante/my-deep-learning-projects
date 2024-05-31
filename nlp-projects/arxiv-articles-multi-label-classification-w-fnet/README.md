## arXiv Articles Multi Label Classification w/FNet

(kaggle link -> https://www.kaggle.com/code/banddaniel/arxiv-articles-multi-label-classification-w-fnet)

<i><span style="color:#e74c3c;"><b>MAIN GOAL: </b> I tried to classify multi-labeled arXiv articles using a FNet. (Mixing Tokens with Fourier Transforms) [1] model.</span></i>


* I applied several <b>preprocessing</b> operations (cleaning,dropping stop words etc.),
* I created <b>multi-class</b> labels,
* Used <b>tf.data pipeline</b> for efficient training,
* I created <b>Vocabulary</b> and trained <b>Tokenizer</b> using the train data,
* 5 KFold <b>cross validation</b>,
* I have modified this notebook [2],
* A function for end-2-end pipeline

## Test Prediction
<img width="558" alt="Screenshot 2024-05-28 at 4 23 58 PM-min-2" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/bf186107-e47a-4857-a732-978aba432363">



## My Another Projects
* [Yelp Review Stars Prediction w/Gemma 7B (LoRA)](https://www.kaggle.com/code/banddaniel/yelp-review-stars-prediction-w-gemma-7b-lora)
* [PaliGemma 3B for License Plate OCR](https://www.kaggle.com/code/banddaniel/paligemma-3b-for-license-plate-ocr)

## References
1. Lee-Thorp, J., Ainslie, J., Eckstein, I., & Ontanon, S. (2021). FNet: Mixing Tokens with Fourier Transforms (Version 4). arXiv. https://doi.org/10.48550/ARXIV.2105.03824
2. https://keras.io/examples/nlp/fnet_classification_with_keras_nlp/
