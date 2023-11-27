# Tomato Detection w/detectron2 (mAP@50: 82.02)

<b> kaggle link -> https://www.kaggle.com/code/banddaniel/tomato-detection-w-detectron2-map-50-82-02 </b>


<span style="color:#e74c3c;"> <b>Note:</b> There was a problem with categorical ids of the test annotation file. I adjusted this problem and uploaded the corrected file.[1] </span> 

I have used the following methods.

* I used a pretrained <b>Faster R-CNN with ResNet-101 Region Proposal Network</b> from detectron2 [2,3]
* I have modified the official notebook [4]


## Test Set Predictions

![280650299-ddb9261e-fa60-4a93-a337-846631b4d4ba](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/aadd7285-29eb-4810-8729-b025e1af7284)
![280650335-660d1480-23aa-46ac-8e69-dbc9b2f7992f](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/98d387b5-62d8-4be3-9370-cdccdfc70497)


## Results

![280650508-0567b8d3-af40-47f8-bf7c-50c7c56b8f96](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/36cda7ec-b3d0-440c-9700-ca000557231e)


## References
1. https://www.kaggle.com/datasets/banddaniel/adjusted-tomato-od-test-json
2. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Version 3). arXiv. https://doi.org/10.48550/ARXIV.1506.01497
3. https://detectron2.readthedocs.io/en/latest/
4. https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
