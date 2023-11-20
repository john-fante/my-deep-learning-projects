# Eye Conjunctiva Segmentation with U-Net

The conjunctiva is a thin membrane in our eyes. It covers the inside of our eyelids and protects our eyes with the mucus layer. This type of segmentation may help quick screening for some diseases like conjunctivitis.

I have used the following methods.

* Dice and Jaccard coefficients implementation,
* The project took place using Google TPU
* Custom layers for encoding and decoding
* Custom callback class  that used predicting a sample from the train dataset during training


## Result
In the 100th epoch
* loss: 0.0024 - dice_coef: 0.9102 - jaccard: 0.8283
* Predictions after each epoch <br>




https://github.com/john-fante/my-deep-learning-projects/assets/50263592/26cc1d20-9c26-4e19-bb9a-acde28413367





## References
1. [Dataset] Rahman, Mohammad Marufur; Faruk, Omar; Ullah, Syed Shah Asheq; Alam, Md. Johurul; Sadman, Shah Md. Safi (2023), “Eye Conjunctiva Segmentation Dataset”, Mendeley Data, V1, doi: 10.17632/yxwjgcndg2.1
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation https://doi.org/10.48550/ARXIV.1505.04597
3. https://en.wikipedia.org/wiki/Conjunctiva
