## Cloud Classification (Involution Neural Network)

(kaggle link -> https://www.kaggle.com/code/banddaniel/cloud-classification-involution-neural-network)


<b>I am very keen on new methods and models in machine learning. This is why I use the involutional neural network [1]. I think it is a very interesting approach. Involution based models are good in respect of the model complexity problem.</b>


I have used the following methods.

* I used an image enhancement method for images (increasing brightness),
* Created an INN (involutional neural network) layer,
* Rectified and recreated functions in this notebook [2],
* Custom callback for evaluating test dataset during training each 10 epochs[3]

<br>

![download (8)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/95ad62ed-7f3b-4e9f-b958-886f8210a237)

<i> <b>Figure 1:</b> Schematic illustration of our proposed involution [1]</i>



## References
1. Li, D., Hu, J., Wang, C., Li, X., She, Q., Zhu, L., Zhang, T., & Chen, Q. (2021). Involution: Inverting the Inherence of Convolution for Visual Recognition (Version 2). arXiv. https://doi.org/10.48550/ARXIV.2103.06255
2. https://keras.io/examples/vision/involution/
3. https://github.com/john-fante/my-tensorflow-custom-callbacks
