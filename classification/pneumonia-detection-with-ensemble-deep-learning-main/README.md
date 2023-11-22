# Pneumonia Detection with Ensemble Deep Learning

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/pneumonia-detection-w-ensemble-dl-0-91-auc </b>

<br>
First of all, I have added samples of both classes to the validation data from the training data. The validation data was very small (only 16 samples). I have used 2 pretrained models, namely Xception and InceptionV3. Lastly, I used an ensemble method using  2 models using  a weighted average layer.<br>

I used followed methods<br>
- Increased the size of the validation set <br>
- Class weights to deal with the class unbalanced problem <br>
- Data augmentation methods <br>
- Two pretrained model <br>
- Weighted average layer <br>

## Results
<br>

- Test Scores <br>

|           | InceptionV3 | Xception | Ensemble Model |
|-----------|-------------|----------|----------------|
| Accuracy  | 84.29 %     | 81.41 %  | 83.17 %        |
| AUC       | 0.9086      | 0.9001   | 0.9120         |
| Precision | 0.8411      | 0.7991   | 0.8188         |
| Recall    | 0.9231      | 0.9385   | 0.9385         |
| Loss      | 0.3736      | 0.3972   | 0.3812         |

- Confusion Matrices <br>

![262597147-a634eafb-99ae-4256-aeb0-e6ea34d3c1df](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/cd256e12-273f-4377-90f1-900d8489bd90)



## References
- https://stackoverflow.com/questions/67647843/is-there-a-way-to-ensemble-two-keras-h5-models-trained-for-same-classes
