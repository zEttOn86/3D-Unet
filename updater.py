#coding:utf-8
"""
@auther tzw
@date 2018-6-15
"""
import os, sys, time
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class Unet3DUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.unet = kwargs.pop("models")
        super(UnetUpdater, self).__init__(*args, **kwargs)

    def loss_softmax_cross_entropy(self, unet, predict, ground_truth):
        """
        * @param unet Unet
        * @param predict Output of unet
        * @param ground_truth Ground truth label
        """
        batchsize = len(predict)
        loss = F.sum(F.log(predict) * ground_truth)/batchsize

        chainer.report({"loss":loss}, unet)#mistery
        return loss

    def update_core(self):
        #load optimizer called "unet"
        unet_optimizer = self.get_optimizer("unet")
        batch = self.get_iterator("main").next()#iterator

        # iterator
        data, label = self.converter(batch, self.device)

        unet = self.unet

        predict = unet(data)

        unet_optimizer.update(self.loss_softmax_cross_entropy, unet, predict, label)
