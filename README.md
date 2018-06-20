# 3D-Unet
Chainer implementation of 3D Unet.  
This program segments brain and unfinished implementaions.  
Training configs are written at coonfigs/base.yml.

## Usage
1. Preprocesing  
Patch coordinate is extracted as csv file.  

    Python preprocessing.py
    
2. Training  
Train 3D unet.  

    Python training.py

## Requirements
- SimpleITK
- pandas  
- Chainer v4
- yaml  

## Umimplemented function
- validation
- test code
- Jaccard index
