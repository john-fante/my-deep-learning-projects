## Chest X-Ray Classification w/ViT (F1 Score: 0.9)

(kaggle link -> https://www.kaggle.com/code/banddaniel/chest-x-ray-classification-w-vit-f1-score-0-9 )

I have used the following methods.

* I used a pretrained <b>ViT</b> architecture for the feature extraction stage [1],
* <b>gelu</b> activation function during the classification stage,
* I used an image processing methods for images <b>(contrast limited adaptive histogram equalization (CLAHE)[2]),</b>
* The project took place using <b>Google TPU</b>,
* Used <b>tf.data</b> for input pipeline,

<br>

<img width="947" alt="download (32)" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/911c390c-5100-4619-925e-c3d3ff31fb16">



## Image Processing Operation

![download (33)](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/870ae5a9-dc0a-443d-9e5d-9ab417aa7277)



## References
1. https://github.com/faustomorales/vit-keras
2. https://en.wikipedia.org/wiki/Adaptive_histogram_equalization
