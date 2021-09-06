# Using fine-tuned Resnet on Cifar10 
![image](https://user-images.githubusercontent.com/55192155/132223216-b5ed712c-a5dc-43e0-8f80-ebb40dfcfce8.png)

# Model
![image](https://user-images.githubusercontent.com/55192155/132225002-e7d93343-39e2-47f3-81bb-6b9e0f52ffd2.png)
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
