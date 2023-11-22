# Higgs/Background Process Classification w/CNN,TPU

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/higgs-background-process-classification-w-cnn-tpu </b>

<br>
First of all, of course, an MLP model was a good option for this problem.<br>
But, I tried to create a CNN model for a binary classification problem using TPU. <br>

- The data has 28 features. I converted the input shape from <b>(None, 28)</b> to <b>(None, 28 ,1)</b> for the convolution operation.
- I split the first 4 tfrecord files of the training dataset for testing. (nearly 1250000 samples)


## Results
<br>

|                 	| Score    	|
|-----------------	|----------	|
| Test AUC Score  	| 0.83351  	|
| Test Precision  	| 0.76832  	|
| Test Recall     	| 0.75736  	|
| Test Accuracy   	| 74.996 % 	|

- Graphs <br>

![261453337-4d29a1ba-b719-43af-8e7c-379f794f7f9d](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/fe505f6b-bc11-434d-8e33-c0850b4d3658)
