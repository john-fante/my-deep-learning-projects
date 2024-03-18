## Crop Disease Classification w/Feature Fusion (EfficientNet, MobileNet)

(kaggle link -> https://www.kaggle.com/code/banddaniel/crop-disease-classify-w-feature-fusion-dl-model)


<i><b><span style="color:#e74c3c;"> Note: Of course, there are other models better than my model in respect of the classification metrics, but I tried to a basic implementation of a paper. </span> </b></i>


First of all, I am very keen on trying new methods. This is why I tried <b>a Future Extractor with Feature Fusion method</b> in this project. I impressed an article named "HCF: A Hybrid CNN Framework for Behavior Detection of Distracted Drivers" in this project [1]. In this technique, first I used 2 pretrained models (EfficientNet v2 and MobileNet v2). At this stage, the 2 models are concatenated with feature outputs (as in the article). Then a dense layer for reducing the dimension as the model output. Finally, I applied PCA and classified with an SVC model. In conclusion, there is a lightly improvement in respect of the model's accuracy and F1 score. 

**I tried to fine-tune the models, but there is no considerable betterment in respect of the F1 score.**


<img width="1190" alt="Screenshot 2024-03-18 at 6 01 32 PM" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/4fa63933-aebd-4ab3-9515-76fc21b27114">

<i>Figure 1: proposed classifier</i>


<br>

* Used <b>mirrored strategy</b>,
* Used <b>tf.data</b> for input pipeline,
* I used an SVC model (with radial basis function kernel) for classification,


## My Another Projects
* [Mammals Classification w/Ensemble Deep Learning](https://www.kaggle.com/code/banddaniel/mammals-classification-w-ensemble-deep-learning)
* [Spam Mail Detection w/Tensorflow (DistilBERT)](https://www.kaggle.com/code/banddaniel/spam-mail-detection-w-tensorflow-distilbert)
* [Towards Data Science Articles Topic Modeling w/LDA](https://www.kaggle.com/code/banddaniel/towards-data-science-articles-topic-modeling-w-lda)


## References
1. Huang, C., Wang, X., Cao, J., Wang, S., & Zhang, Y. (2020). HCF: A Hybrid CNN Framework for Behavior Detection of Distracted Drivers. In IEEE Access (Vol. 8, pp. 109335–109349). Institute of Electrical and Electronics Engineers (IEEE). https://doi.org/10.1109/access.2020.3001159
