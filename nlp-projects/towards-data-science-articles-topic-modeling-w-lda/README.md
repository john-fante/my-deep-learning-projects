## Towards Data Science Articles Topic Modeling w/LDA

(kaggle link -> https://www.kaggle.com/code/banddaniel/towards-data-science-articles-topic-modeling-w-lda)


*I tried to apply <span style="color:#e74c3c;"> Latent Dirichlet Allocation (LDA) </span> model for topic modelling [1].*

* I applied several <b>preprocessing</b> operations (cleaning,dropping stop words, lemmatizing),
* I have chosen <b> 21 for the number of topics</b>. (I tried the topics number value from 5 to 45, but there was no betterment in respect of the Coherence Score.)
* Topic modelling visualization with <b>pyLDAvis</b> library [2].
* An LDA model evaluation with <b>the Coherence Score</b>.


## References
1. https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
2. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
