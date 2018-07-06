#coding:utf-8
"""
@auther tk0103
@date 2018-07-04
"""

from chainer.training import extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
import numpy as np
from chainer import reporter as reporter_module
from chainer.training import extension
import chainer.functions as F

class Unet3DEvaluator(extensions.Evaluator):
    def __init__(self, iterator, unet, number_of_label, eval_func=None, converter=convert.concat_examples, device=None, eval_hook=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self._targets = {'unet':unet}
        self._eval_func = eval_func
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self._max_label = number_of_label

    def loss_softmax_cross_entropy(self, predict, ground_truth):
        eps = 1e-16
        cross_entropy = -F.mean(F.log(predict+eps) * ground_truth)

        return cross_entropy

    def dice_coefficent(self, predict, ground_truth):
        '''
        Assume label 0 is background
        '''
        dice_numerator = 0.0
        dice_denominator = 0.0
        eps = 1e-16

        predict = F.flatten(predict[:,1:self._max_label,:,:,:])
        ground_truth = F.flatten(ground_truth[:,1:self._max_label,:,:,:].astype(np.float32))

        dice_numerator = F.sum(predict * ground_truth)
        dice_denominator =F.sum(predict+ ground_truth)
        dice = 2*dice_numerator/(dice_denominator+eps)

        return dice

    def evaluate(self):
        iterator = self._iterators['main']
        unet = self._targets['unet']
        #eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                ground_truth, data = self.converter(batch, self.device)
                with chainer.using_config("train", False):
                    with chainer.no_backprop_mode():
                        predict = unet(data)
                observation['unet/val/loss'] = self.loss_softmax_cross_entropy(predict, ground_truth)
                observation['unet/val/dice'] = self.dice_coefficent(predict, ground_truth)
            summary.add(observation)

        return summary.compute_mean()
