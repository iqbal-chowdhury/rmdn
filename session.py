# coding: utf-8
from IPython import get_ipython

def init_ipy():
    get_ipython().magic(u'matplotlib inline')
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')

if __name__ == '__main__':
    init_ipy()
