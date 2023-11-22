# Retina Blood Vessel Segmentation w/TPU

I have used the following methods.


* I used two image processing methods for images <span style="color:#e74c3c;"> <b>(Green Channel Conversion[2] , Histogram Equalization[3])</b> </span>
* I used a morphological image processing method for masks <span style="color:#e74c3c;"> <b>(Dilation[4])</b> </span>
* <b>Dice coefficient[5]</b> implementation,
* The project took place using <b>Google TPU</b>,
* <b>Custom layers</b> for encoding and decoding,
* <b>Custom callback</b> class  that used predicting a sample from the test dataset during training
* <b>1000 epochs</b> for training (of course, although this number is very high, the metrics(dice, loss) continued improvement during training)


## My Another Project About Eyes

* [Iris Segmentation U-net w/TPU (Dice Coef: 0.94)](https://www.kaggle.com/code/banddaniel/iris-segmentation-u-net-w-tpu-dice-coef-0-94)
* [Eye Conjunctiva Segmentation with U-Net using TPU (Dice : 0.91, Jaccard : 0.82)](https://github.com/john-fante/eye-conjunctiva-segmentation)


## Test Set Prediction During Training


https://github.com/john-fante/retina-vessel-segmentation/assets/50263592/79b31c76-9d25-415f-84fc-4b1d20f009cf

## Results

![__results___27_1](https://github.com/john-fante/retina-vessel-segmentation/assets/50263592/a254a1eb-4b22-4c41-bf41-a4e998b552ef)

## Test Set Predictions

![download](https://github.com/john-fante/retina-vessel-segmentation/assets/50263592/71709771-382a-401a-9758-58d94ce6a6bb)



## References
1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation (Version 1). arXiv. https://doi.org/10.48550/ARXIV.1505.04597
2. Rathod, Deepali & Manza, Ramesh & Rajput, Yogesh & Patwari, Manjiri & Saswade, Manoj & Deshpande, Neha. (2014). Localization of Optic Disc and Macula using Multilevel 2-D Wavelet Decomposition Based on Haar Wavelet Transform. International Journal of Engineering Research & Technology (IJERT)
3. https://en.wikipedia.org/wiki/Histogram_equalization
4. https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm
5. https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
