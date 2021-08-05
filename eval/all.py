

#!/usr/bin/env python
# _*_ coding:utf-8 _*_


"""
@filename: all.py
@dateTime: 2021-05-29 09:05:21
@author:   unikcc
@contact:  libobo.uk@gmail.com
"""

from main import Pipeline
from attrdict import AttrDict

class Template(object):
    def __init__(self):
        pass

    def forward(self):
        train_rate = [w * 0.1 for w in range(1, 11)]
        for w in train_rate:
            args = AttrDict({'cuda': 0})
            pipeline = Pipeline(args)
            print("Train rate: {:.4f}".format(w))
            pipeline.train_rate = w
            pipeline.main()
            del pipeline

if __name__ == '__main__':
    template = Template()
    template.forward()