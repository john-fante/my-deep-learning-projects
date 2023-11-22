# Zipper Defect Classification (AUC Score: 0.98)

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/zipper-defect-classification-auc-score-0-98 </b>

I have used the following methods.

* I used an image processing methods for images <span style="color:#e74c3c;"> <b>(contrast limited adaptive histogram equalization (CLAHE)[1])</b> </span>
* <b>Custom convolutional block</b> for A CNN model,
* <b>Custom callback</b> class that used creating the test set classification report during training each 50 epochs[2]
* I used the validation set for testing


## Model Architecture
<br>

![270311810-1d92537c-e353-4200-b886-7e43c71e3b28](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/e591d9b2-0dd0-49f8-b60f-3692bc84cac4)


## Results
<br>

![270312193-1407eff7-e1bc-43f8-b852-b550bdd3d5ee](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/1780718e-b06a-43e4-8492-99df46f3dbc5)


## Predictions
<br>
![270312599-776760de-df2c-4405-b93f-f755184ddebf](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/b74db5f9-4176-493a-8037-be0eeddf0aa6)
![270312610-c253fd0d-14bf-4bdf-8252-1726a7066490](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/cf13bccf-0537-47ee-b6b7-474a6ef6d022)


## Confusion Matrix 
<br>

![270313186-9532814e-d5f9-4849-9d10-079f4f64bc21](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/927068be-bc35-4480-9e27-923ae8078bb3)


## References
1. https://en.wikipedia.org/wiki/Adaptive_histogram_equalization
2. My another custom callbacks for Tensorflow (https://github.com/john-fante/my-tensorflow-custom-callbacks)
