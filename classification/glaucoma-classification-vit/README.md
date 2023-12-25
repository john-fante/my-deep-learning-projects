## Glaucoma Classification w/ViT

(kaggle link -> https://www.kaggle.com/code/banddaniel/glaucoma-classification-w-vit-f1-score-0-91)

I have used the following methods.

* I used a pretrained <b>ViT</b> architecture for the feature extraction stage [1],
* <b>gelu</b> activation function during the classification stage,
* I used two image processing methods for images <b>(green channel conversion[2], contrast limited adaptive histogram equalization (CLAHE)[3]),</b>
* The project took place using <b>Google TPU</b>,
* Used <b>tf.data</b> for input pipeline,

<br>


<img width="752" alt="download (2)" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/78ebd611-2190-4668-95dc-31819d9bdd92">

## Image Processing Operation

![download (3)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/00361038-80bf-43e5-a65b-caef19964fc7)


## References
1. https://github.com/faustomorales/vit-keras
2. Rathod, Deepali & Manza, Ramesh & Rajput, Yogesh & Patwari, Manjiri & Saswade, Manoj & Deshpande, Neha. (2014). Localization of Optic Disc and Macula using Multilevel 2-D Wavelet Decomposition Based on Haar Wavelet Transform. International Journal of Engineering Research & Technology (IJERT)
3. https://en.wikipedia.org/wiki/Adaptive_histogram_equalization


