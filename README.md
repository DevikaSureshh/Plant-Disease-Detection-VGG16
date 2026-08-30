# PLANT DISEASE DETECTION USING TRANSFER LEARNING (VGG16)

## PROJECT OVERVIEW

This project presents a plant disease detection system using Transfer Learning with a pre-trained VGG16 deep learning model. The system is developed to classify tomato leaf images into healthy and different disease classes using the PlantVillage dataset.

## OBJECTIVE

The objective of this project is to fine-tune a pre-trained VGG16 model on plant disease images to classify crop leaves as Healthy or Diseased, enabling early agricultural intervention.

## DATASET

The PlantVillage dataset was used for this project. A tomato leaf subset containing 16,011 images belonging to 10 classes was used.

The dataset contains:
- 12,809 training images
- 3,202 validation images
- 10 tomato leaf classes

## TECHNOLOGIES USED

- Python
- TensorFlow/Keras
- VGG16
- NumPy
- Matplotlib
- OpenCV
- Google Colab

## METHODOLOGY

1. Load the PlantVillage tomato leaf dataset.
2. Resize images to 224 × 224 pixels.
3. Normalize the image data.
4. Apply data augmentation using flipping, rotation, and zooming.
5. Load the pre-trained VGG16 model with ImageNet weights.
6. Freeze the VGG16 base layers.
7. Add Global Average Pooling and Dense classification layers.
8. Train the model for 15 epochs.
9. Evaluate the model using accuracy, loss, classification report, and confusion matrix.
10. Unfreeze the last four VGG16 layers.
11. Fine-tune the model and compare the performance.

## RESULTS

The model achieved the following validation accuracy:

| Model | Validation Accuracy |
|---|---:|
| Before Fine-Tuning | 76.51% |
| After Fine-Tuning | 91.63% |

Fine-tuning improved the validation accuracy by **15.12 percentage points**.

## PROJECT RESULTS

The result images, including the confusion matrices, sample predictions, and performance graphs, are available in the `results` folder.

## CONCLUSION

The project demonstrates that transfer learning using VGG16 can effectively classify tomato leaf diseases. Fine-tuning the last four layers significantly improved the model performance from 76.51% to 91.63% validation accuracy. The system can support early identification of plant diseases and help enable timely agricultural intervention.
