#coding:utf-8
"""
* @auther tozawa
* @date 2018-6-15
"""
import os, sys, time
import argparse

from model import UNet3D
from updater import Unet3DUpdater
from dataset import UnetDataset
import util.yaml_utils  as yaml_utils

def main():
    parser = argparse.ArgumentParser(description='Train 3D-Unet')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')
    parser.add_argument('--config_path', type=str, default='configs/base.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results',
                        help='Directory to output the result')
    parser.add_argument('--snapshot', type=str, default='',
                        help='path to the snapshot')
    parser.add_argument('--resume', '-res', default='',
                        help='Resume the training from snapshot')

    parser.add_argument('--training_list', default='configs/training_list.txt',
                        help='Path to training image list file')
    parser.add_argument('--validation_list', default='configs/validation_list.txt',
                        help='Path to validation image list file')
    parser.add_argument('--coordinate_list', type=str, default='configs/coordinate_list.csv')
    args = parser.parse_args()

    '''
    'https://stackoverflow.com/questions/21005822/what-does-os-path-abspathos-path-joinos-path-dirname-file-os-path-pardir'
    '''
    config = yaml_utils.Config(yaml.load(open(os.path.join(os.path.dirname(__file__), args.config_path))))
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(config.batchsize))
    print('# iteration: {}'.format(config.iteration))
    print('')



if __name__ == '__main__':
    main()
