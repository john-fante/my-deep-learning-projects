## Car Detect w/Deep Multiple Instance Learning

(kaggle link -> https://www.kaggle.com/code/banddaniel/car-detect-w-deep-multiple-instance-learning)

I tried the Multiple Instance Learning [1] method in this project. This method is very useful for weakly annotated data and tiled medical images. Actually, this method is not very suitable for this type of binary classification problem and the sample size is very small (due to the sample size, the model has a superior performance in respect to F1 and ROC AUC Scores). However, I only wanted to try the multiple instance learning. 

<span style="color:#e74c3c;"> <b><i> MAIN GOAL: car image detection from a bag that contains both car and bike images </i></b> </span>

<b>If there is at least ONE CAR image, the bag has a positive label (1). If all images are BIKE in the bag, the bag has a negative label (0).</b>

<br>



![download (9)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/fe718c2c-1764-4929-9ec3-732d4a4a8bd0)

<i> <b>Figure 1:</b> A negative label bag example</i>



![download (6)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/d4ba7661-fce1-4c1f-845c-0a6433581918)

<i> <b>Figure 3:</b> Bags using in the Multiple Instance Learning</i>



![download (10)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/c40f2bdd-08bc-4ad8-b524-4d40467fc713)

<i> <b>Figure 4:</b> Attention score results of a test bag   </i>



I have used the following methods.

* I used Gated Attention mechanism from the paper[2],
* Pre-trained ViT model for image features,
* Printing bags examples,
* 2 repetitive training steps (like 2-fold cross validation, but in this case we create new train and validation bags for each loop), at the end, averaging all the 2 loops test predictions,
* Rectified and recreated functions in this notebook [3],
* Printing attention scores, 




## References
1. https://en.wikipedia.org/wiki/Multiple_instance_learning
2. Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning (Version 4). arXiv. https://doi.org/10.48550/ARXIV.1802.04712
3. https://keras.io/examples/vision/attention_mil_classification/
