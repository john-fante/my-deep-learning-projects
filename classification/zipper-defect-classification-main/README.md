# Zipper Defect Classification (AUC Score: 0.98)

I have used the following methods.

* I used an image processing methods for images <span style="color:#e74c3c;"> <b>(contrast limited adaptive histogram equalization (CLAHE)[1])</b> </span>
* <b>Custom convolutional block</b> for A CNN model,
* <b>Custom callback</b> class that used creating the test set classification report during training each 50 epochs[2]
* I used the validation set for testing


## Model Architecture
<br>

![download (4)](https://github.com/john-fante/zipper-defect-classification/assets/50263592/1d92537c-e353-4200-b886-7e43c71e3b28)

## Results
<br>

![__results___24_1](https://github.com/john-fante/zipper-defect-classification/assets/50263592/1407eff7-e1bc-43f8-b852-b550bdd3d5ee)

## Predictions
<br>

![fig_1-min](https://github.com/john-fante/zipper-defect-classification/assets/50263592/776760de-df2c-4405-b93f-f755184ddebf)

![fig_10-min](https://github.com/john-fante/zipper-defect-classification/assets/50263592/c253fd0d-14bf-4bdf-8252-1726a7066490)

## Confusion Matrix 
<br>

![__results___31_1](https://github.com/john-fante/zipper-defect-classification/assets/50263592/9532814e-d5f9-4849-9d10-079f4f64bc21)


## References
1. https://en.wikipedia.org/wiki/Adaptive_histogram_equalization
2. My another custom callbacks for Tensorflow (https://github.com/john-fante/my-tensorflow-custom-callbacks)
