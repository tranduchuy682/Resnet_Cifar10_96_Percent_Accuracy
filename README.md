# Using fine-tuned Resnet on Cifar10 
````
313/313 [==============================] - 1957s 6s/step - loss: 0.0287 - acc: 0.9632
Loss of test model is   0.028715036809444427
Accuracy of test model is  96.31999731063843 %
````
# Model
````
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
up_sampling2d (UpSampling2D) (None, None, None, None)  0         
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, None, None, None)  0         
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, None, None, None)  0         
_________________________________________________________________
resnet50 (Functional)        (None, 8, 8, 2048)        23587712  
_________________________________________________________________
flatten (Flatten)            (None, 131072)            0         
_________________________________________________________________
batch_normalization (BatchNo (None, 131072)            524288    
_________________________________________________________________
dense (Dense)                (None, 128)               16777344  
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 128)               512       
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 64)                256       
_________________________________________________________________
dense_2 (Dense)              (None, 10)                650       
=================================================================
````
- Total params: 40,899,018
- Trainable params: 40,583,370
- Non-trainable params: 315,648

# Loss
- Use BCE loss
# Dataset
- Cifar10
# Training
- Optimizer: RMSprop, with LR: 2e-5, Number of epoch: 60
- Training with batch size: 64
- Data augmentation:
```
datagen = ImageDataGenerator(
    horizontal_flip=flip,
    width_shift_range=width_shift,
    height_shift_range=height_shift,
    rotation_range=15,
    )
```
# Result
- Checkpoint Weight & Model at: https://drive.google.com/file/d/1ZtqDwK5e2vCJFEYfJdp4aYY5NjdkIZI6/view?usp=sharing
- Classification Report
````
              precision    recall  f1-score   support

    airplane       0.98      0.98      0.98      1000
  automobile       0.97      0.98      0.98      1000
        bird       0.96      0.96      0.96      1000
         cat       0.93      0.90      0.91      1000
        deer       0.96      0.96      0.96      1000
         dog       0.94      0.92      0.93      1000
        frog       0.96      0.99      0.97      1000
       horse       0.97      0.98      0.97      1000
        ship       0.98      0.98      0.98      1000
       truck       0.98      0.97      0.97      1000

    accuracy                           0.96     10000
   macro avg       0.96      0.96      0.96     10000
weighted avg       0.96      0.96      0.96     10000
````
- Confusion Matrix
- ![image](https://user-images.githubusercontent.com/55192155/132228661-1e2bcb2f-f19e-4252-8db9-89ebc7a6dcf9.png)
