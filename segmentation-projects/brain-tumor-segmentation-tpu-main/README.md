# Brain tumor/anomaly segmentation with Unet using TPU

This data was obtained from MR images ,then annotated and augmented. Binary masks contain only brain tissue, excluding the tumor tissue.<br>

I have used the following methods.<br>
- Unet architecture with elu activation function<br>
- Dice coefficient implementation<br>
- The project took place using Google TPU<br>

## Results <br>
<br>

- Graphs <br>

![Unknown](https://github.com/john-fante/brain-tumor-segmentation-tpu/assets/50263592/f14118de-e026-4d9d-a5e4-5d8a41dd0f1b)


- Sample prediction masks <br>

![exp1](https://github.com/john-fante/brain-tumor-segmentation-tpu/assets/50263592/eeda0c8e-c962-4176-a4f7-6936338d39ef)

![exp2](https://github.com/john-fante/brain-tumor-segmentation-tpu/assets/50263592/cfd98392-eb05-4b19-8897-9b1ea962bca4)

![exp3](https://github.com/john-fante/brain-tumor-segmentation-tpu/assets/50263592/2760fb93-5ac7-4e0a-9d68-4d4bb8fbfdae)


## References <br>

- Original dataset -> https://www.cancerimagingarchive.net
