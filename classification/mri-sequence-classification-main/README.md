# MRI Sequence Classification w/Custom CNN

## My Goal
I have wanted to build a machine learning project about classifying MRI sequences, because sometimes a specific sequence (for example T2) can be more adequate, for example, a segmentation problem in respect of a research domain. On the other hand, some brain tissues or organs can be visualized in some specific sequence [2].


I have used the following methods.

* I used a custom CNN architecture,
* Splitted train(3582 images) and test set (896 images),
* 5 Kfold cross-validation,
* Used tf.data for input pipeline,


## My Another Projects
* [Rice Classification w/Custom ResNet50 (ACC 85%)](https://www.kaggle.com/code/banddaniel/rice-classification-w-custom-resnet50-acc-85)
* [Lung Segmentation UNet w/SeparableConv (Dice:0.93)](https://www.kaggle.com/code/banddaniel/lung-segmentation-unet-w-separableconv-dice-0-93)
* [Plate Detection w/detectron2 (mAP@75: 89.19)](https://www.kaggle.com/code/banddaniel/plate-detection-w-detectron2-map-75-89-19)


## References
1. https://radiopaedia.org/articles/mri-sequences-overview
2. https://www.weizmann.ac.il/chembiophys/assaf_tal/sites/chemphys.assaf_tal/files/uploads/lecture_8_-_t1_t2_modeling.pdf
