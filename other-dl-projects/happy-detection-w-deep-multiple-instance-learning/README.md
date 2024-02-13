## Happy Detection w/Deep Multiple Instance Learning

(kaggle link -> https://www.kaggle.com/code/banddaniel/happy-detection-w-deep-multiple-instance-learning/)

I tried the Multiple Instance Learning [1] method in this project. This method is very useful for weakly annotated data and tiled medical images. Actually, this method is not very suitable for this type of binary classification problem and the sample size is very small (due to the sample size, the model has a superior performance in respect to F1 and ROC AUC Scores). However, I only wanted to try the multiple instance learning. 


<span style="color:#e74c3c;"> <b><i> MAIN GOAL: a HAPPY face detection from a bag that contains all classes </i></b> </span>

<b>If there is at least ONE HAPPY image, the bag has a positive label (1). If all images are not HAPPY in the bag, the bag has a negative label (0).</b>

<br>

![output-onlinepngtools-2-2](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/97ba06eb-1993-48c3-93c2-fb779f397703)


<i> <b>Figure 1:</b> A negative label bag example</i>


<br>

![output-onlinepngtools-2](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/b67094e8-453c-4d2d-9805-16adec36f8ce)

<i> <b>Figure 2:</b> A positive label bag example</i>

<br>


![download](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/965de469-3a7f-418f-9989-7c2f56db33d2)

<i> <b>Figure 3:</b> Bags using in the Multiple Instance Learning</i>


<br>

![output-onlinepngtools-3-2](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/09b5d1c5-a40f-44e8-bf24-b228a2748a2c)

<i> <b>Figure 4:</b> Attention score results of a test bag   </i>



I have used the following methods.

* I used Gated Attention mechanism from the paper[2],
* ResNet101 model for image features,
* Printing bags examples,
* 2 repetitive training steps (like 2-fold cross validation, but in this case we create new train and validation bags for each loop), at the end, averaging all the 2 loops test predictions,
* Rectified and recreated functions in this notebook [3],
* Printing attention scores, 


## References
1. https://en.wikipedia.org/wiki/Multiple_instance_learning
2. Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning (Version 4). arXiv. https://doi.org/10.48550/ARXIV.1802.04712
3. https://keras.io/examples/vision/attention_mil_classification/
