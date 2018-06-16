#coding:utf-8
'''
* @auther mygw
* @date 2018-6-15
'''

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import HeNormal as w

class UNet3D(chainer.Chain):

    def __init__(self, label):
        super(UNet3D, self).__init__()
        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=3,in_channels=1, out_channels=32, ksize=3)
            self.conv2 = L.ConvolutionND(ndim=3,in_channels=32, out_channels=64, ksize=3)

            self.conv3 = L.ConvolutionND(ndim=3,in_channels=64, out_channels=64, ksize=3)
            self.conv4 = L.ConvolutionND(ndim=3,in_channels=64, out_channels=128, ksize=3)

            self.conv5 = L.ConvolutionND(ndim=3,in_channels=128, out_channels=128, ksize=3)
            self.conv6 = L.ConvolutionND(ndim=3,in_channels=128, out_channels=256, ksize=3)

            self.conv7 = L.ConvolutionND(ndim=3,in_channels=256, out_channels=256, ksize=3)
            self.conv8 = L.ConvolutionND(ndim=3,in_channels=256, out_channels=512, ksize=3)

            self.dconv1 = L.DeconvolutionND(ndim=3, in_channels=512, out_channels=512, ksize=2, stride=2)
            self.conv9 = L.ConvolutionND(ndim=3,in_channels=256 + 512, out_channels=256, ksize=3)
            self.conv10 = L.ConvolutionND(ndim=3,in_channels=256, out_channels=256, ksize=3)

            self.dconv2 = L.DeconvolutionND(ndim=3, in_channels=256, out_channels=256, ksize=2, stride=2)
            self.conv11 = L.ConvolutionND(ndim=3,in_channels=128 + 256, out_channels=128, ksize=3)
            self.conv12 = L.ConvolutionND(ndim=3,in_channels=128, out_channels=128, ksize=3)

            self.dconv3 = L.DeconvolutionND(ndim=3, in_channels=128, out_channels=128, ksize=2, stride=2)
            self.conv13 = L.ConvolutionND(ndim=3,in_channels=64 + 128, out_channels=64, ksize=3)
            self.conv14 = L.ConvolutionND(ndim=3,in_channels=64, out_channels=64, ksize=3)
            self.conv15 = L.ConvolutionND(ndim=3,in_channels=64, out_channels=label, ksize=1)

            self.bnc0=L.BatchNormalization(32)
            self.bnc1=L.BatchNormalization(64)
            self.bnc2=L.BatchNormalization(64)
            self.bnc3=L.BatchNormalization(128)
            self.bnc4=L.BatchNormalization(128)
            self.bnc5=L.BatchNormalization(256)
            self.bnc6=L.BatchNormalization(256)
            self.bnc7=L.BatchNormalization(512)
            # bnc8=L.BatchNormalization(512)

            # bnd9=L.BatchNormalization(512)
            self.bnd8=L.BatchNormalization(256)
            self.bnd7=L.BatchNormalization(256)
            # bnd6=L.BatchNormalization(256)
            self.bnd5=L.BatchNormalization(128)
            self.bnd4=L.BatchNormalization(128)
            # bnd3=L.BatchNormalization(128)
            self.bnd2=L.BatchNormalization(64)
            self.bnd1=L.BatchNormalization(64)
            self.train = True



    def __call__(self, x):
        test = not self.train

        h1 = F.relu(self.bnc0(self.conv1(x)))
        h2 = F.relu(self.bnc1(self.conv2(h1)))
        #cover_all please check map size
        h3 = F.max_pooling_nd(h2, ksize=2, stride=2)
        del h1

        h4 = F.relu(self.bnc2(self.conv3(h3)))
        del h3
        h5 = F.relu(self.bnc3(self.conv4(h4)))
        del h4
        h6 = F.max_pooling_nd(h5,ksize=2,stride=2)

        h7 = F.relu(self.bnc4(self.conv5(h6)))
        del h6
        h8 = F.relu(self.bnc5(self.conv6(h7)))
        del h7
        h9 = F.max_pooling_nd(h8, ksize=2,stride=2)


        h10 = F.relu(self.bnc6(self.conv7(h9)))
        del h9
        h11 = F.relu(self.bnc7(self.conv8(h10)))
        del h10

        h12 = self.dconv1(h11)
        del h11
        h13 = F.concat([h12, self.cropping(h8,h12)], axis=1)
        del h8, h12

        h14 = F.relu(self.bnd8(self.conv9(h13)))
        del h13
        h15 = F.relu(self.bnd7(self.conv10(h14)))
        del h14

        h16 = self.dconv2(h15)
        del h15
        h17 = F.concat([h16, self.cropping(h5,h16)])
        del h5, h16
        h18 = F.relu(self.bnd5(self.conv11(h17)))
        del h17
        h19 = F.relu(self.bnd4(self.conv12(h18)))
        del h18

        h20 = self.dconv3(h19)
        del h19
        h21 = F.concat([h20, self.cropping(h2,h20)])
        del h2, h20

        h22 = F.relu(self.bnd2(self.conv13(h21)))
        del h21
        h23 = F.relu(self.bnd1(self.conv14(h22)))
        del h22

        h24 = F.softmax(self.conv15(h23), axis=1) #probility
        del h23

        return h24


    def cropping(self, input, ref):

        edgez = (input.shape[2] - ref.shape[2])/2
        edgey = (input.shape[3] - ref.shape[3])/2
        edgex = (input.shape[4] - ref.shape[4])/2
        edgez = int(edgex)
        edgey = int(edgey)
        edgex = int(edgez)


        X = F.split_axis(input,(edgex,int(input.shape[4]-edgex)),axis=4)
        X = X[1]
        X = F.split_axis(X,(edgey,int(X.shape[3]-edgey)),axis=3)
        X = X[1]
        X = F.split_axis(X,(edgez,int (X.shape[2]-edgez)),axis=2)
        X=X[1]
        return X
