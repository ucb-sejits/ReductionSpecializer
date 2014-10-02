"""
specializer XorReduction
"""

from __future__ import print_function

import numpy as np
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.frontend import get_ast
from ctree import browser_show_ast
from ctree.transformations import PyBasicConversions
from ctree.nodes import CtreeNode
from ctree.c.nodes import For, SymbolRef, Assign, Lt, PostInc, \
    Constant, Deref
import ctree.np
import ctypes as ct

import ast

from collections import namedtuple

class XorReductionFrontend(PyBasicConversions):
    pass

class Slimmy(LazySpecializedFunction):

    subconfig_type = namedtuple('subconfig',['dtype','ndim','shape','flags'])

    def __init__(self, py_ast = None):
        py_ast = py_ast or get_ast(self.kernel)
        super(Slimmy, self).__init__(py_ast)

    def args_to_subconfig(self, args):
        A = args[0]
        return Slimmy.subconfig_type(A.dtype, A.ndim, A.shape,[])


    def transform(self, tree, program_config):
        browser_show_ast(tree,'tmp.png')
        arg_cfg, tune_cfg = program_config
        tree = XorReductionFrontend().visit(tree)
        tree = XorReductionCBackend(arg_cfg).visit(tree)
        fn = ConcreteXorReduction()
        arg_type = np.ctypeslib.ndpointer(*arg_cfg)
        print(tree.files[0])
        return fn.finalize('kernel',
                           tree,
                           ct.CFUNCTYPE(None,arg_type, ct.POINTER(ct.c_int32))
                           )

    def points(self, inpt):
        return np.nditer(inpt, flags=['c_index'])

class XorReduction(Slimmy):
    def kernel(self, inpt):
        '''
            Calculates the cumulative XOR of elements in inpt, equivalent to
            Reduce with XOR
        '''
        result = 0
        for point in self.points(inpt):
            result ^= point
        return result



if __name__ == '__main__':
    XorReducer = XorReduction()
    arr = (np.random.rand(1024*1024)*100).astype(np.int32)
    print(XorReducer(arr))

