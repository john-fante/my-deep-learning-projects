## Brain Tumor Segmentation (detectron2, mAP@50:76.2)

(kaggle link -> https://www.kaggle.com/code/banddaniel/brain-tumor-segmentation-detectron2-map-50-76-2)


I have used the following methods.

* I used a pretrained <b>Mask R-CNN with ResNeXt-101-32x8d for Feature Pyramid Network</b> from detectron2 [1,2]
* I have modified the official notebook [3]
* I used validation and test sets for testing,

## Test Predictions

<img width="946" alt="Screenshot 2024-02-05 at 11 51 51 AM" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/16da23eb-e7f0-4e7d-8429-b2ee079d0d88">
<img width="905" alt="Screenshot 2024-02-05 at 11 52 02 AM" src="https://github.com/john-fante/my-deep-learning-projects/assets/50263592/a6c1dcc2-d13e-438e-9942-b2c28b273ffb">



## References
1. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Version 3). arXiv. https://doi.org/10.48550/ARXIV.1506.01497
2. https://detectron2.readthedocs.io/en/latest/
3. https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
