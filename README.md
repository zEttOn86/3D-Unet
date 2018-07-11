# 3D-Unet
Chainer implementation of 3D Unet for brain segmentaion.  
Training configs are written at coonfigs/base.yml.  
Because of the limitaion of GPU memory, we used patch based method.

## Requirements
- SimpleITK
- pandas  
- Chainer v4
- yaml  

## Usage
__Training__  
To train 3D unet.  
```
Python training.py
```    
__Prediction__  
To segment images with trained network.  
```
Python prediction.py
```

## Training result
Training loss and dice score.
![loss](https://github.com/zEttOn86/3D-Unet/blob/master/results/training/unet_loss.png)  
![dice](https://github.com/zEttOn86/3D-Unet/blob/master/results/training/unet_dice_score.png)

