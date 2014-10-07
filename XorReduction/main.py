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

class PointsLoop(CtreeNode):
    _fields = ['target', 'iter_target', 'body']

    def __init__(self, target, iter_target, body):
        self.target = target
        self.iter_target = iter_target
        self.body = body
        #super(PointsLoop, self).__init__()

    def label(self):
        return str(self.iter_target)

class XorReductionFrontend(PyBasicConversions):
    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and \
            isinstance(node.iter.func, ast.Attribute) and \
            node.iter.func.attr is 'points' and \
            node.iter.func.value.id is 'self':
            target = node.target.id
            iter_target = node.iter.args[0].id
            body = [self.visit(statement) for statement in node.body]
            return PointsLoop(target, iter_target, body)
        else:
            return node

class XorReductionCBackend(ast.NodeTransformer):
    def __init__(self, arg_cfg):
        self.arg_cfg = arg_cfg
        self.retval = None

    def visit_FunctionDecl(self, node):

        # what happens if you have multiple args?
        arg_type = np.ctypeslib.ndpointer(self.arg_cfg.dtype, self.arg_cfg.ndim, self.arg_cfg.shape)  # arg_type is the c-type of the input (like int *)

        # Get the actual params
        param = node.params[1]                                      # note that params[0] is self
        param.type = arg_type()                                     # because logic
        node.params = [param]                                       # this basically strips out "self"

        ## Alternative to above code
        # node.params[1].type = argtype()
        # node.params = node.params[1:]

        ## Adding the 'output' variable as one of the parameters of type argtype
        retval = SymbolRef("output", arg_type())                    # retval is the "output" of type argtype
        self.retval = "output"                                      # "output" is the name of 
        node.params.append(retval)                                  # this appends the output parameter to the list of parameters
        node.defn = list(map(self.visit, node.defn))
        node.defn[0].left.type = arg_type._dtype_.type()
        return node

    def visit_PointsLoop(self, node):
        target = node.target
        return For(
            Assign(SymbolRef(target, ct.c_int()), Constant(0)),
            Lt(SymbolRef(target), Constant(self.arg_cfg.size)),
            PostInc(SymbolRef(target)),
            list(map(self.visit, node.body))
        )

    def visit_Return(self, node):
        return Assign(Deref(SymbolRef(self.retval)), node.value)


class ConcreteXorReduction(ConcreteSpecializedFunction):
    def finalize(self, entry_name, tree, entry_type):
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, inpt):
        output = ct.c_int()
        self._c_function(inpt, ct.byref(output))
        return output.value    
    


class LazySlimmy(LazySpecializedFunction):

    subconfig_type = namedtuple('subconfig',['dtype','ndim','shape','size','flags'])

    def __init__(self, py_ast = None):
        py_ast = py_ast or get_ast(self.kernel)
        super(Slimmy, self).__init__(py_ast)

    def args_to_subconfig(self, args):
        # what happens if you have more than one arg?
        A = args[0]
        return Slimmy.subconfig_type(A.dtype, A.ndim, A.shape, A.size, [])


    def transform(self, tree, program_config):
        #browser_show_ast(tree,'tree_init.png')
        arg_cfg, tune_cfg = program_config
        tree = XorReductionFrontend().visit(tree)
        #browser_show_ast(tree, 'tree_post_frontend.png')
        tree = XorReductionCBackend(arg_cfg).visit(tree)
        #browser_show_ast(tree, 'tree_post_backend.png')
        fn = ConcreteXorReduction()
        arg_type = np.ctypeslib.ndpointer(arg_cfg.dtype, arg_cfg.ndim, arg_cfg.shape, arg_cfg.flags)
        print(tree.files[0])
        return fn.finalize('kernel',
                           tree,
                           ct.CFUNCTYPE(None,arg_type, ct.POINTER(ct.c_int32))
                           )

    def points(self, inpt):
        return np.nditer(inpt)




class XorReduction(LazySlimmy):
    def kernel(self, inpt):
        '''
            Calculates the cumulative XOR of elements in inpt, equivalent to
            Reduce with XOR
        '''
        result = 0
        for point in self.points(inpt):
            result = point ^ result
        return result



if __name__ == '__main__':
    XorReducer = XorReduction()
    arr = np.array([0b1100,0b1001])
    print(reduce(lambda x,y: x^y, np.nditer(arr), 0))
    print(XorReducer(arr))

