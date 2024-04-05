## BBC News Topic Modeling w/LDA

(kaggle link -> https://www.kaggle.com/code/banddaniel/bbc-news-topic-modeling-w-lda)

*I tried to apply <span style="color:#e74c3c;"> Latent Dirichlet Allocation (LDA) </span> model for topic modelling [1].*

* I applied several <b>preprocessing</b> operations (cleaning,dropping stop words, lemmatizing),
* I have chosen <b> 20 for the number of topics</b>. (I tried the topics number value from 5 to 45, but there was no betterment in respect of the Coherence Score.)
* Topic modelling visualization with <b>pyLDAvis</b> library [2].
* An LDA model evaluation with <b>the Coherence Score</b>.

## Results
* Coherence Score: 0.4362164


## My Another Projects
* [Manufacturing Question-Answer w/Fine-Tuning Gemma 7B (LoRA)](https://www.kaggle.com/code/banddaniel/manufacturing-question-answer-w-gemma-7b-lora)
* [News Analysis w/Tensorflow (DistilBERT)](https://www.kaggle.com/code/banddaniel/news-analysis-w-tensorflow-distilbert)
* [Complaint Analysis w/Ensemble Model (CatBoost, LR)](https://www.kaggle.com/code/banddaniel/complaint-analysis-w-ensemble-model-catboost-lr)


## References
1. https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
2. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
