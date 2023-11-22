# Brain tumor/anomaly segmentation with Unet using TPU

This data was obtained from MR images ,then annotated and augmented. Binary masks contain only brain tissue, excluding the tumor tissue.<br>

I have used the following methods.<br>
- Unet architecture with elu activation function<br>
- Dice coefficient implementation<br>
- The project took place using Google TPU<br>

## Results <br>
<br>

- Graphs <br>

![265419086-f14118de-e026-4d9d-a5e4-5d8a41dd0f1b](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/5909830b-07c8-4c01-bec9-d380a38d7a71)


- Sample prediction masks <br>

![265419374-eeda0c8e-c962-4176-a4f7-6936338d39ef](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/365a8033-f493-49bd-b0c0-1a1af0102b2b)



## References <br>

- Original dataset -> https://www.cancerimagingarchive.net
