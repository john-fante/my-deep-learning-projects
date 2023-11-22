# Brain tissue segmentation with Unet using TPU

This data was obtained from MR images ,then annotated and augmented. Binary masks contain only brain tissue, excluding the tumor tissue.<br>

I have used the following methods.<br>
- Unet architecture with elu activation function<br>
- Dice coefficient implementation<br>
- The project took place using Google TPU<br>

## Results <br>
- This model have achived 0.88 validation Dice Coefficient.

<br>

- Graphs <br>


![Unknown](https://github.com/john-fante/brain-tissue-segmentation-tpu/assets/50263592/b1f68434-d5aa-4665-ba1e-4c2c11e931cd)

- Sample prediction masks <br>

![Unknown-4](https://github.com/john-fante/brain-tissue-segmentation-tpu/assets/50263592/d180fbb2-2580-45a0-81ec-ad6f771a7fd4)

![Unknown-3](https://github.com/john-fante/brain-tissue-segmentation-tpu/assets/50263592/25caca15-e4c1-4e8f-903a-e9382b79ff99)

![Unknown-5](https://github.com/john-fante/brain-tissue-segmentation-tpu/assets/50263592/88f0d0ec-fd38-4d32-bb15-94978a54ff37)

- Model <br>

![Unknown-2](https://github.com/john-fante/brain-tissue-segmentation-tpu/assets/50263592/55141f72-7426-4a31-be31-8bdae5483554)


## References <br>
- Original dataset -> https://www.cancerimagingarchive.net


