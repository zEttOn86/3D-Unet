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
<img src="https://github.com/zEttOn86/3D-Unet/blob/master/results/training/unet_loss.png" alt="loss" title="loss" width=70% height=70%>  
<img src="https://github.com/zEttOn86/3D-Unet/blob/master/results/training/unet_dice_score.png" alt="dice" title="dice" width=70% height=70%>

## Predicted result
Example of input image  
<img src="https://github.com/zEttOn86/3D-Unet/blob/master/results/prediction/input_image.png" alt="input" title="input" width=50% height=50%>  
Example of ground truth  
<img src="https://github.com/zEttOn86/3D-Unet/blob/master/results/prediction/ground_truth.png" alt="gt" title="gt" width=50% height=50%>  
Example of prediction  
<img src="https://github.com/zEttOn86/3D-Unet/blob/master/results/prediction/prediction.png" alt="p" title="p" width=50% height=50%>  

## Results
We calculated jaccard index  

| label | J.I. |
| :---: | :---: |
| 0 | 0.99553 |
| 1 | 0.83438 |
| 2 | 0.86771 |
| 3 | 0.91392 |
| 4 | 0.80850 |
| 5 | 0.88321 |
| 6 | 0.87240 |
