#coding:utf-8
import os, sys, time
import argparse
import SimpleITK as sitk
import numpy as np

def jaccard(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)

    if float(intersection.sum()) == 0.:
        return 0.
    else:
        return intersection.sum() / float(union.sum())


def main():
    parser = argparse.ArgumentParser(description='Calc jaccard index')
    parser.add_argument('--inputImageFile', '-i', help='Input image file')
    parser.add_argument('--groundTruthImageFile', '-g', help='Ground truth image file')
    args = parser.parse_args()

    sitkInput = sitk.ReadImage(args.inputImageFile)
    input = sitk.GetArrayFromImage(sitkInput).flatten()
    sitkGroundTruth = sitk.ReadImage(args.groundTruthImageFile)
    gt = sitk.GetArrayFromImage(sitkGroundTruth).flatten()
    number_of_labels = np.unique(gt)
    print('Number of labels: {}'.format(number_of_labels))

    for label in number_of_labels:
        ans_label = np.array([1 if i==label else 0 for i in gt])
        input_label = np.array([1 if i==label else 0 for i in input])
        ji = jaccard(ans_label, input_label)

        print('Label: {} Jaccard index = {}'.format(label, ji))


if __name__ == '__main__':
    main()
