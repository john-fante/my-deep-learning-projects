## Social Media Post Multiclass Classification w/DistilBERT

(kaggle link -> https://www.kaggle.com/code/banddaniel/social-media-post-multiclass-classify-w-distilbert)


<i><span style="color:#e74c3c;"><b>MAIN GOAL: </b>  I tried to fine-tune a DistilBERT model for solving a multi-class problem (multi output model). The first prediction is bias class (neutral or partisan), the second  prediction is message (policy, personal, support, information etc.). </span></i>


* I applied several <b>preprocessing</b> operations (cleaning,dropping stop words),
* Used <b>tf.data pipeline</b> for efficient training,
* I only used only 75 max length for sequence length (BERT models support up to 512 input lengths),


## My Another Projects
* [Complaint Analysis w/Ensemble Model (CatBoost, LR)](https://www.kaggle.com/code/banddaniel/complaint-analysis-w-ensemble-model-catboost-lr)
* [Gemma 2B Text Summarization w/Zero-Shot Prompting](https://www.kaggle.com/code/banddaniel/gemma-2b-text-summarization-w-zero-shot-prompting)
* [Crop Disease Classification w/Feature Fusion, DL Model](https://www.kaggle.com/code/banddaniel/crop-disease-classify-w-feature-fusion-dl-model)
