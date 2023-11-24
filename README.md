**Convolutional Neural Network Using PyTorch that can translate photos of ASL letters to text.**
- Training dataset used is https://www.kaggle.com/datasets/grassknoted/asl-alphabet
  
The NN can be trained by running
```
python3 utils/training.py <number of training images to use per category>
```
This saves a .pth model file in the models/ folder that can be imported and used for testing.
