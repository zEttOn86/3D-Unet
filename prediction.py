#coding:utf-8

'''
* @auther mygw
* @date 2018-6-15
'''

import numpy as np
import cupy as cp

import argparse, os

import chainer 
import chainer.functions as F
from chainer import Variable
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer.cuda import to_cpu
from chainer.cuda import to_gpu

import SimpleITK as sitk

from tqdm import tqdm
import math

from model import UNet3D as unet


def cropping_patch(image, coordx, coordy, coordz, patchx,patchy,patchz,xp):
    x = xp.zeros(patchx*patchy*patchz).reshape(1,1,patchz,patchy,patchx)


    x[0][0] = image[0,0,int(coordz):int(coordz+patchz),int(coordy):int(coordy+patchy),int(coordx):int(coordx+patchx)]
 
    return (x,t)

def constructing_image(image, patch,coordx,coordy,coordz,patchx,patchy,patchz):

    image[int(coordz):int(coordz+patchz),int(coordy):int(coordy+patchy),int(coordx):int(coordx+patchx)] = patch
    return image

def load_text(filepath):
    file_name = []
    print("load " + filepath ) 
    with open(filepath) as f:
        all_line = f.readlines()
        for line in all_line:
            file_name.append(line.replace("\n",""))  
    return file_name






def main():

    parse = argparse.ArgumentParser(description='Prediction 3D U-Net')
    parser.add_argument('--root', '-R', default=os.path.direname(os.path.abspath(__file__)), 
                        help='Root directory path of input image')
    parser.add_argument('--out','-o', default=os.path.direname(os.path.abspath(__file__)),
                        help='Directory to output the result')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default=os.path.dirname(os.path.abspath(__file__))+'/model.npz',
                        help='Load model data')
    parser.add_argument('--casetext', '-c', default=os.path.dirname(os.path.abspath(__file__))+'case_text.txt',
                        help='Text writen test cases')
    parser.add_argument('--patchx', '-x', default=100,
                        help='size of x width of intput patch')
    parser.add_argument('--patchy', '-y', default=100,
                        help='size of y width of intput patch')
    parser.add_argument('--patchz', '-z',default=100,
                        help='size of z width of intput patch')
    parser.add_argument('--margin', default=0,
                        help='the difference size of one side between input and output by convolution')
    parser.add_argument('--label', '-l', default=2,
                        help='Class number')
    args = parser.parse_args()


    print('GPU: {}'.format(argparse.gpu))
    print('patchsize: ({},{},{})'.format(argparse.patchx,argparse.patchy,argparse.patchz))
    print('')
    
    xp = np
    use_cudnn = False
    gpu_id = argparse.gpu
    seg = unet.UNet3D(argparse.label)

    if(gpu_id> 0):
        xp = cp
        use_cudnn = True
        seg.to_gpu
    else:
        seg.to_cpu

    serializers.load_npz(argparse.model)


    test_text = load_text(argparse.casetext)


    for n in range(len(test_text)):

        print("load {}".format(test_text[n]))

        Image3D = sitk.ReadImage(argparse.out + test_text[n] + ".mhd")
        t_image = sitk.GetArrayFromImage(Image3D)
        

        patchxo = argparse.patchx-argparse.margin
        patchyo = argparse.patchy-argparse.margin
        patchzo = argparse.patchz-argparse.margin



        (org_sizex,org_sizey,org_sizez) = t_image.GetSize()

        y_org = np.zeros(org_sizex*org_sizey*org_sizez).reshape(org_sizex,org_sizey,org_sizez)
        element_num = argparse.label * y_org.size

        y_org_soft = np.zeros(element_num,dtype=np.float32).reshape(argparse.label, org_sizex+patchxo, org_sizey+patchyo, org_sizez+patchzo)

        xe = int(math.ceil(org_sizex/patchxo))
        ye = int(math.ceil(org_sizey/patchyo))
        ze = int(math.ceil(org_sizez/patchzo))

        t_image = np.pad(t_image,argparse.margin,'reflect')
        xn = xp.array(t_image[None,None, ...],xp.float32)


        pbar = tqdm(total = xe*ye*ze)
        for s in range(xe*ye*ze):
            xi = int(s % xe)
            yi = int((s % (ye*xe)) / xe)
            zi = int(s / (ye*xe))
            x = cropping_patch(xn,tn,xi*patchxo,yi*patchyo,zi*patchzo,patchx,patchy,patchz,xp)

            x = Variable(x.astype(xp.float32))

            with chainer.using_config("train", False):
                y = seg(x)

            y_soft = F.softmax(y, axis=1)
            for j in range(y.shape[1]):
                if(gpu_id>0):
                    y_ = to_cpu(y_soft.data[0,j,:,:,:])
                else:
                    y_ = y_soft.data[0,j,:,:,:]    
                y_ = np.array(y_,np.float32)

                y_org_soft[j] = constructing_image(y_org_soft[j],y_,xi*patchxo,yi*patchyo,zi*patchzo,patchxo,patchyo,patchzo)


            if(gpu_id>0):
                y_out = np.array(to_cpu((F.argmax(y_soft,axis = 1)).data))
            else:
                y_out = np.array((F.argmax(y_soft,axis = 1)).data)

            y_org = constructing_image(y_org,y_out,xi*patchxo,yi*patchyo,zi*patchzo,patchxo,patchyo,patchzo)


            pbar.update()

        
        y_org_soft = y_org_soft[:,:int(org_sizez),:int(org_sizey),:int(org_sizex)]
        y_org = y_org[:int(org_sizez),:int(org_sizey),:int(org_sizex)]    

        
        ImageUC = sitk.GetImageFromArray(y_org)
        ImageUC.SetSpacing((Image3D.GetSpacing))
        ImageUC.SetSpacing((Image3D.GetOrigin))
        sitk.WriteImage(ImageUC, os.path.join(argparse.out, "estimation", test_text[n] + ".mhd"))

        for i in range(y_org_soft.shape[0]):
            ImageD = sitk.GetImageFromArray(y_org_soft)
            ImageD.SetSpacing((Image3D.GetSpacing))
            ImageD.SetOrigin((Image3D.GetOrigin))
            sitk.WriteImage(ImageD, os.path.join(argparse.out, "softmax",  test_text[n], "ch" + str(i) + ".mhd"))
        
        pbar.close()

    print("complete")



if __name__ == '__main__':
    main()
        
