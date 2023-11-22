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

![263110418-b1f68434-d5aa-4665-ba1e-4c2c11e931cd](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/633397d4-abe3-4c3c-a4a7-8fbe06fa9385)


- Sample prediction masks <br>

![263110756-d180fbb2-2580-45a0-81ec-ad6f771a7fd4](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/b43fa926-e29e-4d21-b08f-68d872ec3bdb)

![263111042-25caca15-e4c1-4e8f-903a-e9382b79ff99](https://github.com/john-fante/my-deep-learning-projects/assets/50263592/dc7f85d7-0c18-4e48-a56b-e32113121328)


- Model <br>

![Uploading 263111208-55141f72-7426-4a31-be31-8bdae5483554.png…]()



## References <br>
- Original dataset -> https://www.cancerimagingarchive.net


