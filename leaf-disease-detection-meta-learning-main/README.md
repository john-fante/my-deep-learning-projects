# Leaf Disease Detection w/Meta-Learning (ViT, PCA, SVM)

I tried a meta-learning method in this project. In this technique, first I used a pretrained ViT (Vision Transformer) model for the feature extraction stage, then applied PCA for the curse of dimensionality problem, and finally used a tuned SVC model for the classification stage.


### ViT Model (for Feature Extraction) -> PCA (for Dimensionality Reduction) -> SVC (for Classification)


|                                 | Training Feature Shape |
|---------------------------------|-------------|
| ViT Features                    | (974, 64)  |
| After PCA (98 % Variance Ratio) | (974, 41)   |


I have used the following methods.

* I tried to implementation of distributed deep learning strategy,
* I split the full data into train (974 images), validation (133 images) and test (60 images),
* I used a pretrained ViT model [1],
* Used <b>tf.data</b> for input pipeline,
* I used a SVM model for classification (tuned with optuna),

## Test Results
![__results___28_1](https://github.com/john-fante/leaf-disease-detection-meta-learning/assets/50263592/2b0f68b1-07e2-4a45-bcd9-00103d616ff0)

## Test Predictions
![__results___31_0](https://github.com/john-fante/leaf-disease-detection-meta-learning/assets/50263592/791a46bb-5336-4dc2-ab28-350250aeca7c)
![__results___31_2](https://github.com/john-fante/leaf-disease-detection-meta-learning/assets/50263592/83a626b6-b98f-4796-b71f-8ed9f825a8db)

## References
1. https://github.com/faustomorales/vit-keras
