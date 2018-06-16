#coding:utf-8
'''
* @auther tzw
* @date 2018-6-15
'''
import os, sys, time
import numpy as np
import chainer
import pandas as pd
import util.dataIO as IO

class UnetDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, data_list_txt, coordinate_list_csv, patch_side):
        print(' Initilaze dataset ')
        self._root = root
        self._patch_side = patch_side
        self._max_label = 7 #[0, 7)

        assert(self._patch_side%2==0)

        """
        * Read path to org and label data
        hogehoge.txt
        org.mhd org_label.mhd
        """
        path_pairs = []
        with open(data_list_txt) as paths_file:
            for line in paths_file:
                line = line.split()
                if not line : continue
                path_pairs.append(line[:])

        self._num_of_case = len(path_pairs)
        print('    # of cases: {}'.format(self._num_of_case))

        self._dataset=[]
        for i in path_pairs:
            print('   Org   from: {}'.format(i[0]))
            print('   label from: {}'.format(i[1]))
            # Read data
            org = IO.read_mhd_and_raw(os.path.join(self._root, 'data', i[0])).astype("float32")
            org = org[np.newaxis, :]#(ch, z, y, x)
            label_ = IO.read_mhd_and_raw(os.path.join(self._root, 'data', i[1])).flatten()
            label = np.zeros((org.shape[1]*org.shape[2]*org.shape[3], self._max_label), dtype=int)
            # one-hot encoding
            #"https://stackoverflow.com/questions/29831489/numpy-1-hot-array"
            label[np.arange(org.shape[1]*org.shape[2]*org.shape[3]), label_] = 1
            label = label.transpose().reshape(self._max_label, org.shape[1], org.shape[2], org.shape[3])
            self._dataset.append((org, label))

        """
        * Read coordinate csv
        fugafuga.csv
        x1,y1,z1
        x2,y2,z2
        ...
        xn,yn,zn
        """
        self._coordinate = pd.read_csv(os.path.join(self._root, coordinate_list_csv),names=("x","y","z")).values.tolist()

        print(' Initilazation done ')

    def __len__(self):
        return (int)(len(self._coordinate))

    def get_example(self, i):
        '''
        return (label, org)

        I assume the same number of patches can be extracted in all training images
        '''
        case_number = int(i % self._num_of_case)
        x,y,z=self._coordinate[i]
        x_s, x_e = (x - int(self._patch_side/2)), (x + int(self._patch_side/2))
        y_s, y_e = (y - int(self._patch_side/2)), (y + int(self._patch_side/2))
        z_s, z_e = (z - int(self._patch_side/2)), (z + int(self._patch_side/2))

        return self._dataset[case_number][1][:, z_s:z_e, y_s:y_e, x_s:x_e], self._dataset[case_number][0][:, z_s:z_e, y_s:y_e, x_s:x_e]
