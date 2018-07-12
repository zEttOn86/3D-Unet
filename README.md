# 3D-Unet
Chainer implementation of 3D Unet for brain segmentaion.  
Training configs are written at coonfigs/base.yml.  
Because of the limitaion of GPU memory, we used patch based method.

## Requirements
- SimpleITK
- Chainer v4
- yaml  

## Usage
__Training__  
To train 3D unet.  
```
python train.py -h

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU, -g GPU     GPU ID (negative value indicates CPU)
  --base BASE, -B BASE  base directory path of program files
  --config_path CONFIG_PATH
                        path to config file
  --out OUT, -o OUT     Directory to output the result
  --model MODEL, -m MODEL
                        Load model data
  --resume RESUME, -res RESUME
                        Resume the training from snapshot
  --root ROOT, -R ROOT  Root directory path of input image
  --training_list TRAINING_LIST
                        Path to training image list file
  --validation_list VALIDATION_LIST
                        Path to validation image list file
```  

Example:  
To train using gpu
```
python train.py -g 0
```  

__Prediction__  
To segment images with trained network.  
```
python predict.py -h

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU, -g GPU     GPU ID (negative value indicates CPU)
  --base BASE, -B BASE  base directory path of program files
  --config_path CONFIG_PATH
                        path to config file
  --out OUT, -o OUT     Directory to output the result
  --model MODEL, -m MODEL
                        Load model data(snapshot)
  --root ROOT, -R ROOT  Root directory path of input image
  --test_list TEST_LIST
                        Path to test image list file
```
  
Example:  
To predict 
```
python predict.py -g 0 -m results/training/UNet3D_150000.npz
```

## Training result
Training loss and dice score.
<img src="https://github.com/zEttOn86/3D-Unet/blob/master/results/training/unet_loss.png" alt="loss" title="loss" width=50% height=50%>
![loss](https://github.com/zEttOn86/3D-Unet/blob/master/results/training/unet_loss.png)  
![dice](https://github.com/zEttOn86/3D-Unet/blob/master/results/training/unet_dice_score.png)

## Predicted result
Example of input image
![input](https://github.com/zEttOn86/3D-Unet/blob/master/results/prediction/input_image.png)  
Example of ground truth
![gt](https://github.com/zEttOn86/3D-Unet/blob/master/results/prediction/ground_truth.png)  
Example of prediction
![p](https://github.com/zEttOn86/3D-Unet/blob/master/results/prediction/prediction.png)  

## Future plan
We have some plan to add evaluation code to measure Jaccard index.

