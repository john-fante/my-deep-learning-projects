# Down Syndrome Detection with CNN

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/down-syndrome-detection-w-cnn-test-auc-0-878 </b>

<br>
I used followed methods<br>

* A custom CNN model with 194,529 trainable parameters
* Splitted train(2549 images) and test set (450 images) <br>
* Used <b>elu</b> activation function during feature extraction
* Used tf.data for input pipeline

## CNN Model Architecture

![265772843-ec20f806-6dbc-4a02-96f9-afbca5855630](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/03b14361-4cef-4283-95d9-8923247a67d9)


## Results
<br>

|                | Result  |
|----------------|---------|
| Test Accuracy  | 80.44 % |
| Test AUC       | 0.878   |
| Test Precision | 0.862   |
| Test Recall    | 0.735   |

* Graphs
  
![265772945-31faee4b-1e83-49e5-9ac4-e9827faf4c16](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/97183ac0-6811-4f31-8e64-835ed127dbbb)


* Confusion Matrix

![265773118-778fb198-e0d8-4dfc-88b5-62a6cdf2eb92](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/d6efadf1-29a0-45be-862d-879d26511ce4)
