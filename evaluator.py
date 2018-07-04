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

class UNet3DEvaluator(extensions.Evaluator):
    def __init__(self, iterator, unet,eval_func=None,converter=convert.concat_examples,device=None, eval_hook=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self._targets = {'unet':unet}
        self._eval_func = eval_func
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook

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
                ground_truth,data = self.converter(batch, self.device)
                predict = unet(data)
                eps = 1e-16
                loss = -F.mean(F.log(predict+eps) * ground_truth)
                observation['vali/unet/loss'] = loss
            summary.add(observation)

        return summary.compute_mean()
