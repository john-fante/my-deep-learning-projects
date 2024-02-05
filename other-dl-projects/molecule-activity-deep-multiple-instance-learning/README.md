## Molecule Activity, Deep Multiple Instance Learning

(kaggle link -> https://www.kaggle.com/code/banddaniel/molecule-activity-deep-multiple-instance-learning)

<b>I tried the Multiple Instance Learning [1] method in this project. This method is very useful for weakly annotated data and tiled medical images.</b>

I have used the following methods.

* I used Gated Attention mechanism from the paper[2],
* Printing bags examples,
* 3 repetitive training steps (like 3-fold cross validation, but in this case we create new train and validation bags for each loop), at the end, averaging all the 3 loops test predictions
* Rectified and recreated functions in this notebook [3],
* Printing attention scores, 

 
<br>

![download (6)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/c46088a8-5ff4-4fd2-b7ad-0fe17670e764)

<i> <b>Figure 1:</b> Bags using in the Multiple Instance Learning</i>

<br>


![download (7)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/6612e71f-1216-4ed0-8290-09f220721dfe)

<i> <b>Figure 2:</b> Proposed Deep Multiple Instance Learning pipeline with gated attention [2]</i>



## References
1. https://en.wikipedia.org/wiki/Multiple_instance_learning
2. Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning (Version 4). arXiv. https://doi.org/10.48550/ARXIV.1802.04712
3. https://keras.io/examples/vision/attention_mil_classification/
