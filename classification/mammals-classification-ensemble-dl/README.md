## Mammals Classification w/Ensemble Deep Learning

First of all, I am very keen on trying new methods. This is why I tried an Ensemble Deep Learning method in this project. I impressed an article named "HCF: A Hybrid CNN Framework for Behavior Detection of Distracted Drivers" in this project [1]. In this technique, first I used 2 pretrained models (Xception, and DenseNet201). Then saved models and weights for using in the ensemble model. At this stage, I combined the 2 models with the GlobalAveragePooling2D layer outputs (as in the article). In conclusion, there is a lightly improvement in respect of the model's accuracy and F1 score.

<i><b><span style="color:#e74c3c;"> Note: Of course, there are other models better than my model in respect of the classification metrics, but I tried to a basic implementation of a paper. </span> </b></i>

## Xception + DenseNet201 --> The Ensemble Model

![download (35)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/75467368-195a-4ec3-a53a-8131002572c8)

<i> Figure : the proposed framework in the paper [1]</i>



I have used the following methods.

* The project took place using <b>Google TPU</b>,
* Used <b>tf.data</b> for input pipeline,
* I split the full data into train, validation and test sets,
* Used <b>tf.data</b> for input pipeline,


## References
1. Huang, C., Wang, X., Cao, J., Wang, S., & Zhang, Y. (2020). HCF: A Hybrid CNN Framework for Behavior Detection of Distracted Drivers. In IEEE Access (Vol. 8, pp. 109335–109349). Institute of Electrical and Electronics Engineers (IEEE). https://doi.org/10.1109/access.2020.3001159
