#coding:utf-8
'''
* @auther tzw
* @date 2018-7-10
'''

import os, sys, time
import argparse, yaml, shutil, math
import numpy as np
import chainer
import SimpleITK as sitk

from model import UNet3D
import util.yaml_utils  as yaml_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='configs/base.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results/prediction',
                        help='Directory to output the result')

    parser.add_argument('--model', '-m', default='',
                        help='Load model data(snapshot)')

    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')
    parser.add_argument('--test_list', default='configs/test_list.txt',
                        help='Path to test image list file')
    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))
    print('GPU: {}'.format(args.gpu))
    print('')

    unet = UNet3D(config.unet['number_of_label'])
    chainer.serializers.load_npz(args.model, unet)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        unet.to_gpu()
    xp = unet.xp

    # Read test list
    path_pairs = []
    with open(os.path.join(args.base, args.test_list)) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line : continue
            path_pairs.append(line[:])

    for i in path_pairs:
        print('   Org   from: {}'.format(i[0]))
        print('   label from: {}'.format(i[1]))
        sitkOrg = sitk.ReadImage(os.path.join(args.root, 'data', i[0]))
        org = sitk.GetArrayFromImage(sitkOrg).astype("float32")

        # Calculate maximum of number of patch at each side
        ze,ye,xe = org.shape
        xm = int(math.ceil((float(xe)/float(config.patch['patchside']))))
        ym = int(math.ceil((float(ye)/float(config.patch['patchside']))))
        zm = int(math.ceil((float(ze)/float(config.patch['patchside']))))

        margin = ((0, config.patch['patchside']),
                  (0, config.patch['patchside']),
                  (0, config.patch['patchside']))
        org = np.pad(org, margin, 'edge')
        org = chainer.Variable(xp.array(org[np.newaxis, np.newaxis, :], dtype=xp.float32))

        prediction_map = np.zeros((ze+config.patch['patchside'],ye+config.patch['patchside'], xe+config.patch['patchside']))
        probability_map = np.zeros((config.unet['number_of_label'], ze+config.patch['patchside'], ye+config.patch['patchside'], xe+config.patch['patchside']))

        # Patch loop
        for s in range(xm*ym*zm):
            xi = int(s%xm)*config.patch['patchside']
            yi = int((s%(ym*xm))/xm)*config.patch['patchside']
            zi = int(s/(ym*xm))*config.patch['patchside']
            # Extract patch from original image
            patch = org[:,:,zi:zi+config.patch['patchside'],yi:yi+config.patch['patchside'],xi:xi+config.patch['patchside']]
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                probability_patch = unet(patch)

            # Generate probability map
            probability_patch = probability_patch.data
            if args.gpu >= 0:
                probability_patch = chainer.cuda.to_cpu(probability_patch)
            for ch in range(probability_patch.shape[1]):
                probability_map[ch,zi:zi+config.patch['patchside'],yi:yi+config.patch['patchside'],xi:xi+config.patch['patchside']] = probability_patch[0,ch,:,:,:]

            prediction_patch = np.argmax(probability_patch, axis=1)

            prediction_map[zi:zi+config.patch['patchside'],yi:yi+config.patch['patchside'],xi:xi+config.patch['patchside']] = prediction_patch[0,:,:,:]

        print('Save image')
        probability_map = probability_map[:,:ze,:ye,:xe]
        prediction_map = prediction_map[:ze,:ye,:xe]

        # Save prediction map
        imagePrediction = sitk.GetImageFromArray(prediction_map)
        imagePrediction.SetSpacing(sitkOrg.GetSpacing())
        imagePrediction.SetOrigin(sitkOrg.GetOrigin())
        result_dir = os.path.join(args.base, args.out, os.path.dirname(i[0]))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        fn = os.path.splitext(os.path.basename(i[0]))[0]
        sitk.WriteImage(imagePrediction, '{}/{}.mhd'.format(result_dir, fn))

        # Save probability map
        for ch in range(probability_map.shape[0]):
            imageProbability = sitk.GetImageFromArray(probability_map)
            imageProbability.SetSpacing(sitkOrg.GetSpacing())
            imageProbability.SetOrigin(sitkOrg.GetOrigin())
            sitk.WriteImage(imageProbability, '{}/{}_probability_{}.mhd'.format(result_dir, fn, ch))


if __name__ == '__main__':
    main()
