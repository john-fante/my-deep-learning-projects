## Yelp Review Stars Prediction w/Gemma 7B (LoRA)

(kaggle link -> https://www.kaggle.com/code/banddaniel/yelp-review-stars-prediction-w-gemma-7b-lora)

<i><span style="color:#e74c3c;"><b>MAIN GOAL: </b>  I tried to predict (classify) review stars with a fine-tuned Gemma 7B model using prompts. 
</span></i>

<i><span style="color:#e74c3c;"><b>NOTE: </b>  There was an issue with the Gemma model that returned the given prompt identically (almost 5% of all test predictions). I tried to solve this problem by choosing a random star. </span></i>
 

<b>Example Prompt: <i> Give this review a rating of 0 to 5, just one rating, no explanation. + REVIEW </i></b>


* I applied several <b>preprocessing</b> operations (cleaning,dropping stop words etc.),
* I created <b>Gemma prompts</b>,
* I fine-tuned a Gemma 7B model with <b>LoRA</b>,
* I have modified this notebook [1],
* An end-2-end test prediction pipeline function,
* Test evaluation metrics <b>(f1 score, classification report etc.)</b>,


## Test Predictions

<img width="1243" alt="Screenshot 2024-04-11 at 10 04 23 PM" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/fe5d576b-d844-4399-bec0-338ddc046ea6">

<img width="1240" alt="Screenshot 2024-04-11 at 10 04 13 PM" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/dd4ad3ec-3016-4b70-87cb-e9bc347d9f81">


## References
1. https://ai.google.dev/gemma/docs/lora_tuning
