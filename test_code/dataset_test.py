#coding:utf-8
import os, sys, time
import argparse
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))
from dataset import UnetDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', type=str, default='F:/project/3D-Unet',
                        help='Directory to input image')
    parser.add_argument('--traing_list', type=str, default='configs/training_list.txt')
    parser.add_argument('--coordinate_list', type=str, default='configs/coordinate_list.csv')

    args = parser.parse_args()

    UnetDataset(root=args.root,
                data_list_txt=args.traing_list,
                coordinate_list_csv=args.coordinate_list,
                patch_side=3)

if __name__ == '__main__':
    main()
