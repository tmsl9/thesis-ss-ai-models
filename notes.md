It was firstly tested 3 semantic segmentation AI models: Mask R-CNN, U-Net and YOLOv8n.
Some experiments were done with the models:
- epochs: 50 and 75;
- learning rate: 0.0001 only;
- training image size: 480, 640 and 720;
- total images: 45 images almost with the same size, and 63 which is the 45 images plus other 18 images with different resolutions;
- pre-processing: none, gaussian filter and/or grayscale;
- data augmentations: no and flip horizontal and vertical 2 times.

The best results are with:
- whatever epochs number, 50 and 75 is already robust numbers;
- whatever training image size;
- 45 images which means images have all the same resolution;
- gaussian filter or without the results are very similar;
- data augmentations, it robusts the dataset since the dataset has a short amount of images, improving then the results;
- Mask R-CNN and U-Net have the best results out of the 3 models tested, with accuracy higher than 83 %, recall higher than 81 % and F1 score higher than 84 %. YOLOv8n doesn't reach even 80 % of accuracy, the best results being 76 %, 76.8 % and 76.4 % of precision, recall and F1 score respectively.

The best result is acc=0.852, recall=0.854 and F1-score=0.853, with the following model configuration:
- model: Mask R-CNN;
- Epochs: 75;
- Learning rate: 0.0001;
- Training image size: 480;
- Dataset: 45 images;
- Pre-processing: none;
- Data augmentations: flip Horizontal Vertical 2x.
However, the second best result has basically the same metrics values and the model used only 50 epochs. Therefore this configuration was instead used in order to fast the experiments time, since there are lots of use-cases to test out. 

Next, it was used the best result configuration in order to test all the pre-processing techniques noticed in the related works. The techniques are the following:
- Adaptive median filter with smoothing level 3, 5 and 7 (the higher the smoother);
- Background masking or region of interest which I assume it's the same since the region of interest goal on pre-processing phase and not in augmentation process is to only show the lesion areas;
- Bilateral smooth filter;
- Binary mask;
- Color histogram;
- Contrast enhancement;
- Dilation;
- Erosion;
- Gaussian filter;
- Grayscale;
- Hair removal;
- Region of interest;
- Resize;
- Sharpening.

Notes:
- Adaptive median filter and texture detection techniques work with grayscale inputs, which means that the output images will also be in grayscale.
- Hair removal, depite of the paper BLAH BLAH which proposes a deep learning method to remove the hairs from an image, I don't have access to IEEE as student.
The paper compares their proposed model with the DullRazor implementation which only uses segmenation and no AI is involded: https://github.com/BlueDokk/Dullrazor-algorithm. So this one is used instead for testing. The Dullrazor algorithm applies a gaussian filter in the images.

Until now, out of 72 runs with the previously best configuration (model, epochs, learning rate and training image size) the best results are:
| POS | MODEL | EPOCHS | LEARNING_RATE | TRAINING_IMGSIZE | TOTAL_IMAGES | PREPROCESSING | AUGMENTATIONS | PREDICT_IMAGES | PRECISION | RECALL | F1_SCORE | PREDICT_TIME_PER_IMAGE |
|-----|-----------|--------|---------------|------------|------------|------------------------------------|------------------------------|-----|-----------|--------|----------|-------|
| 1   | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | bilateral smooth filter 15                      | flipHorizontalVertical & 2x | 5   | 0.89  | 0.852 | 0.871 | 0.192 |
| 2   | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | adaptive median filter 3                        | no                          | 5   | 0.831 | 0.9   | 0.864 | 0.002 |
| 3   | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | resize 720x720                                  | no                          | 5   | 0.845 | 0.88  | 0.862 | 0.132 |
| 4   | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | dilation                                        | no                          | 5   | 0.893 | 0.829 | 0.86  | 0.262 |
| 5   | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | resize 640x640 & hair removal & gaussian filter | flipHorizontalVertical & 2x | 5   | 0.893 | 0.828 | 0.859 | 0.174 |

The conclusions to take are:
- Resizing is the best technique to use, since it occupies 60 % of the top 5 F1 score board above. Generalizing the image size seems to help but why?;
- Using bilateral smooth filter with minimal smoothing (preserving more details) appears once out of 5. This is explained because all the details of the picture are kept, and there are more texture details to be extracted from the model and learned. We can see that the best results are the ones where the picture details are more preserved.
- Not using any pre-processing technique gives good results as well, it occupies the second position of the top 5. This is explained because all the details of the picture are kept, and there are more texture details to be extracted from the model and learned. We can see that the best results are the ones where the picture details are more preserved.

Going through the pre-processing techniques, it was tested each one of the techniques alone, and the best result for each is:
| POS | MODEL | EPOCHS | LEARNING_RATE | TRAINING_IMGSIZE | TOTAL_IMAGES | PREPROCESSING | AUGMENTATIONS | PREDICT_IMAGES | PRECISION | RECALL | F1_SCORE | PREDICT_TIME_PER_IMAGE |
|-----|-----------|--------|---------------|------------|------------|------------------------------------|------------------------------|-----|-----------|--------|----------|-------|
| 1   | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | resize 640x640                     | no                           | 5   | 0.822     | 0.936  | 0.875    | 0.369 |
| 2   | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | no                                 | flipHorizontalVertical & 2x  | 5   | 0.885     | 0.785  | 0.832    | 0.853 |
| 4   | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | bilateral smooth filter 5 50 50    | flipHorizontalVertical & 2x  | 5   | 0.7       | 0.973  | 0.814    | 0.161 |
| 5   | Mask RCNN | 50     | 0.0001        | 480x480    | 63         | resize 720x720                     | flipHorizontalVertical & 2x  | 7   | 0.815     | 0.747  | 0.78     | 0.301 |
| 6   | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | resize 480x480                     | no                           | 5   | 0.906     | 0.669  | 0.77     | 0.364 |
| 8   | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | grayscale                          | flipHorizontalVertical & 2x  | 5   | 0.661     | 0.894  | 0.76     | 0.133 |
| 12  | Mask RCNN | 50     | 0.0001        | 480x480    | 63         | erosion                            | flipHorizontalVertical & 2x  | 7   | 0.659     | 0.844  | 0.74     | 0.181 |
| 14  | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | bilateral smooth filter 9 75 75    | no                           | 5   | 0.582     | 0.986  | 0.732    | 0.962 |
| 16  | Mask RCNN | 50     | 0.0001        | 480x480    | 63         | dilation                           | no                           | 7   | 0.743     | 0.695  | 0.718    | 0.152 |
| 23  | Mask RCNN | 50     | 0.0001        | 480x480    | 63         | bilateral smooth filter 15 100 100 | no                           | 7   | 0.648     | 0.734  | 0.688    | 0.091 |
| 34  | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | contrast enhancement               | no                           | 5   | 0.834     | 0.431  | 0.568    | 0.177 |
| 35  | Mask RCNN | 50     | 0.0001        | 480x480    | 63         | color histogram                    | flipHorizontalVertical & 2x  | 7   | 0.853     | 0.421  | 0.564    | 0.079 |
| 40  | Mask RCNN | 50     | 0.0001        | 480x480    | 63         | gaussian filter                    | no                           | 7   | 0.512     | 0.557  | 0.534    | 0.092 |
| 45  | Mask RCNN | 50     | 0.0001        | 480x480    | 63         | hair removal                       | no                           | 7   | 0.442     | 0.578  | 0.501    | 0.771 |
| 50  | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | sharpening                         | no                           | 5   | 0.945     | 0.259  | 0.407    | 0.416 |
| 55  | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | adaptive median filter 3           | flipHorizontalVertical & 2x  | 5   | 0.158     | 0.981  | 0.272    | 0.113 |
| 64  | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | adaptive median filter 5           | flipHorizontalVertical & 2x  | 5   | 0.0       | 0.0    | 0.0      | 0.135 |
| 66  | Mask RCNN | 50     | 0.0001        | 480x480    | 45         | adaptive median filter 7           | flipHorizontalVertical & 2x  | 5   | 0.0       | 0.023  | 0.0      | 0.172 |

Conclusions:
- Techniques which degrade or remove the most of the texture details of the image have the worst results.
- Only the techniques that preserve the most the details have good or reasonable results.

But, we should considerate that the model will receive pictures with different resolutions. So, checking the experiments with the dataset with images with different sizes, the top 5 results are:
| POS | MODEL | EPOCHS | LEARNING_RATE | TRAINING_IMGSIZE | TOTAL_IMAGES | PREPROCESSING | AUGMENTATIONS | PREDICT_IMAGES | PRECISION | RECALL | F1_SCORE | PREDICT_TIME_PER_IMAGE |
|-----|-----------|--------|---------------|------------|------------|------------------------------------|------------------------------|-----|-----------|--------|----------|-------|
| 5   | Mask RCNN | 50     | 0.0001        | 480x480    | 63         | resize 720x720                     | flipHorizontalVertical & 2x  | 7   | 0.815     | 0.747  | 0.78     | 0.301 |
| 9   | Mask RCNN | 50     | 0.0001        | 480x480    | 63         | bilateral smooth filter 5 50 50    | no                           | 7   | 0.711     | 0.827  | 0.765    | 0.109 |
| 10  | Mask RCNN | 50     | 0.0001        | 480x480    | 63         | no                                 | no                           | 7   | 0.789     | 0.73   | 0.758    | 0.753 |
| 12  | Mask RCNN | 50     | 0.0001        | 480x480    | 63         | bilateral smooth filter 5 50 50    | flipHorizontalVertical & 2x  | 7   | 0.675     | 0.834  | 0.746    | 0.112 |
| 15  | Mask RCNN | 50     | 0.0001        | 480x480    | 63         | erosion                            | flipHorizontalVertical & 2x  | 7   | 0.659     | 0.844  | 0.74     | 0.181 |

In this more realistic case, the results are not so good. As we can see, this top 5 is in reality starting on 5th and ends on 15th of the total rank grid. Only the best case reaches 80 % of accuracy but F1 score doesn't reach that value. We can see that the best results are the ones where the picture details are more preserved.

Now it's time to replicate the test the pre-processing techniques that the articles used together.

| Technique | Article [1] | Article [2] | Article [5] | Article [6] | Article [8] | Article [12] | Article [13] | Article [15] | Article [18] | Article [19] | Article [20] | Article [21] | Article [22] | Article [23] |
|-------------------------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Color histogram         | + | - | - | - | - | - | - | - | - | - | - | - | - | - |
| Resize                  | - | + | + | - | + | - | - | + | - | - | + | + | - | + |
| Hair removal            | - | + | - | - | - | - | - | - | - | - | - | - | - | - |
| Gaussian filter         | - | + | - | + | - | - | - | - | - | - | + | - | + | + |
| Adaptive median filter  | - | - | + | - | - | - | - | - | - | - | - | - | - | - |
| Grayscale               | - | - | + | - | - | - | + | - | - | + | - | - | - | - |
| Median filter           | - | - | - | + | - | - | - | - | - | + | - | - | - | - |
| Contrast enhancement    | - | - | - | + | - | + | - | - | - | - | - | - | - | - |
| Region of interest      | - | - | - | + | - | - | - | - | - | - | - | - | - | - |
| Background masking      | - | - | - | - | + | - | - | - | - | - | - | - | - | - |
| Noise removal*          | - | - | - | - | - | - | - | - | + | - | - | - | - | - |
| Sharpening              | - | - | - | - | - | - | - | - | - | + | - | - | - | - |
| Bilateral smooth filter | - | - | - | - | - | - | - | - | - | + | - | - | - | - |
| Binary mask             | - | - | - | - | - | - | - | - | - | + | - | - | - | - |
| Erosion                 | - | - | - | - | - | - | - | - | - | - | - | - | + | - |
| Dilation                | - | - | - | - | - | - | - | - | - | + | - | - | + | - |



NOTE:
    - Why article 18 has "binary mask" as pre-processing technique? I will take this as ground truth measure. It doesn't seem to be background masking as well because their explanation for the technique usage is:
        "Binary Mask
            Each pixel in a binary image is represented as a single bit, either a 0 or a 1. Bit 1 indicates image pixels that belong to the object, while 0 indicate image pixels of the background."


======================================================================
TODO:
    - chapter 3:
        Add text and table: "Considering that the model will encounter images of varying resolutions in practical scenarios, the TABLE NR 3 shows the top results of the set of experiments performed on a dataset with images of different sizes."
    - in the conclusion, mention that the model configuration should use a training image size and/or resize dimensions more close to reality. Meaning, I would need to check but at least my Samsung S21 takes picture >3000x>3000 resolution.