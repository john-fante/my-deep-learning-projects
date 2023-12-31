## Brain Tumor Classification (Normal, Glioma, Meningioma, Pituitary)

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/mri-classification-86-275-test-acc </b>

<br>
❗️As far as I am concerned, there is a problem with the data. <br>

<br>The problem stems from the different image planes.
There are three planes in MR images called sagittal, axial, and coronal [1].
However, the dataset contains images from all three planes, and this situation can reduce the model's robustness.
Although there is a problem, I yielded plausible accuracy and  an auc score using a basic architecture. 
<br>


<img style="width:70%; margin:10px;" src = "https://github.com/john-fante/my-deep-learning-projects/assets/50263592/ac25d804-eacd-460d-83a3-52ffa1473d4c" alt="mri planes" />


## Result
<p>Of course, this type of big fluctuation in validation metrics is not good. Changing some callbacks criteria help improve the model generalization. In my experience, sometimes the accuracy balances when epochs increase. </p>
<p>Test accuracy : 86.275 %</p>

## References
1. https://www.researchgate.net/publication/304891093_A_Study_of_MRI_Segmentation_Methods_in_Automatic_Brain_Tumor_Detection 
