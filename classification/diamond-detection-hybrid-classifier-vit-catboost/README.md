## Diamond Detect w/Hybrid Model (ViT,CatBoost,SHAP)

(kaggle link -> https://www.kaggle.com/code/banddaniel/diamond-detect-w-hybrid-model-vit-catboost-shap )

I tried to use a hybrid model in this project. In this technique, first I used a custom ViT (Vision Transformer) model for the feature extraction stage,  then merged the metadata .csv file with the ViT features, then applied PCA for the curse of dimensionality problem, and finally used a CatBoost model for the classification stage.



### <span style="color:#e74c3c;">  ViT Model (for Feature Extraction) -> Merging with .csv file (using for another features) -> PCA (for Dimensionality Reduction) -> CatBoostClassifier (for Classification) </span> 


|                                 | Training Feature Shape |
|---------------------------------|-------------|
| ViT Features                    | (39464, 64)  |
| After PCA (99 % Variance Ratio) | (39464, 42)   |



* I used a mirrored strategy (using 2 T4 GPU at the same time),
* I split the full data into train (39498 images), validation (4389 images) and test (4877 images),
* I used a customized ViT model [1],
* Used <b>tf.data</b> for input pipeline,
* I used a CatBoost model for classification,
* SHAP for feature explanation,


## References
1. https://github.com/faustomorales/vit-keras
