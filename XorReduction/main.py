"""
specializer XorReduction
"""

from __future__ import print_function

import logging

logging.basicConfig(level=20)


import numpy as np
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.frontend import get_ast
from ctree import browser_show_ast
from ctree.transformations import PyBasicConversions
from ctree.nodes import CtreeNode, Project
from ctree.c.nodes import For, SymbolRef, Assign, Lt, PostInc, \
    Constant, Deref, FunctionDecl, Add, Mul, If, And, ArrayRef, FunctionCall, \
    CFile, Eq, Mod, AugAssign, MulAssign
import ctree.np
import ctypes as ct
from ctree.ocl.macros import get_global_id, get_group_id, get_local_id
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
import pycl as cl

from math import log



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

        # TODO: stripping out self, as is done above, should really be abstracted, as it's the same everywhere
        # Get the actual params (stripping out "self" from node.params)
        param = node.params[1]                                      # note that params[0] is self
        param.type = arg_type()                                     # because logic
        node.params = [param]                                       # this basically strips out "self"

        ## Alternative to above code; stripping out "self" from node.params
        # node.params[1].type = argtype()
        # node.params = node.params[1:]

        # TODO: below, 'output' should really be (int *) hardcoded, rather than the same
        #       as the input type (which is represented by argtype)
        ## Adding the 'output' variable as one of the parameters of type argtype
        retval = SymbolRef("output", arg_type())                # retval is a symbol reference to c-variable named "output" of type argtype
        self.retval = "output"                                  # 'output' is the name of
        node.params.append(retval)                              # this appends the output parameter to the list of parameters
        node.defn = list(map(self.visit, node.defn))
        node.defn[0].left.type = arg_type._dtype_.type()
        return node

    def visit_PointsLoop(self, node):
        target = node.target
        return For(
            # TODO: Not sustainable... what happens i starts at 1?
            Assign(SymbolRef(target, ct.c_int()), Constant(0)),   # int i = 0;
            Lt(SymbolRef(target), Constant(self.arg_cfg.size)),   # 'Lt' = Less than; i < size of array
            PostInc(SymbolRef(target)),                           # i++
            list(map(self.visit, node.body))                      # Recursively call the other nodes
        )

    def visit_Return(self, node):
        return Assign(Deref(SymbolRef(self.retval)), node.value)  # *output = <return value>


class ConcreteXorReduction(ConcreteSpecializedFunction):

    def __init__(self):
        self.context = cl.clCreateContextFromType()
        self.queue = cl.clCreateCommandQueue(self.context)

    def finalize(self, kernel, tree, entry_name, entry_type):
        self.kernel = kernel
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, A):
        buf, evt = cl.buffer_from_ndarray(self.queue, A, blocking=False)
        self._c_function(self.queue, self.kernel, buf)
        B, evt = cl.buffer_to_ndarray(self.queue, buf, like=A)
        return B



class LazySlimmy(LazySpecializedFunction):

    subconfig_type = namedtuple('subconfig',['dtype','ndim','shape','size','flags'])

    def __init__(self, py_ast = None):
        py_ast = py_ast or get_ast(self.apply)
        super(LazySlimmy, self).__init__(py_ast)

    def args_to_subconfig(self, args):
        # what happens if you have more than one arg?
        A = args[0]
        return LazySlimmy.subconfig_type(A.dtype, A.ndim, A.shape, A.size, [])


    def transform(self, tree, program_config):
        
        A = program_config[0]
        len_A = np.prod(A.shape)
        inner_type = A.dtype.type()
        pointer = np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape)
        apply_one = PyBasicConversions().visit(tree.body[0])
        apply_one.return_type = inner_type
        apply_one.params[0].type = inner_type
        apply_one.params[1].type = inner_type



        # C-Code for in-place operations
        # __kernel void apply_kernel(__global float* A) {
        #     int i = get_global_id(0);
        #     A[i] = apply(A[i])
        # };

        # C-Code for Reduction
        # __kernel void apply_kernel(__global int* A) {
        #     int i = get_global_id(0);
        #     if (i + 1 < <Array length>) {
        #         A[i] = apply(A[i], A[i+1]);
        #     };
        # };
        # __kernel void apply_kernel(__global int* A)
        # {
        #     int i = get_global_id(0);
        #     for (int iter = 0; iter < <Arr Len>; iter *= 2)
        #         {
        #             if (i % 2*iter == 0)
        #                 {
        #                     A[i] = apply(A[i], A[i+2**iter]);
        #                 }
        #         }
        # }
        apply_kernel = FunctionDecl(None, "apply_kernel",
                                    params=[SymbolRef("A", pointer()).set_global()],
                                    defn=[
                                        Assign(SymbolRef('groupId', ct.c_int()), get_group_id(0)),
                                        Assign(SymbolRef('id',ct.c_int()), get_global_id(0)),
                                        For(Assign(SymbolRef('i', ct.c_int()), Constant(1)), Lt(SymbolRef('i'), Constant(32)),
                                            MulAssign(SymbolRef('i'), Constant(2)),
                                            [
                                                If(Eq(Mod(SymbolRef('id'),Mul(SymbolRef('i'),Constant(2))), Constant(0)),
                                                   [
                                                       Assign(ArrayRef(SymbolRef('A'),SymbolRef('id')),
                                                              FunctionCall(SymbolRef('apply'),
                                                                           [
                                                                               ArrayRef(SymbolRef('A'), SymbolRef('id')),
                                                                               ArrayRef(SymbolRef('A'), Add(SymbolRef('id'),SymbolRef('i')))
                                                                           ])),
                                                   ]
                                                ),
                                                FunctionCall(SymbolRef('barrier'),[SymbolRef('CLK_LOCAL_MEM_FENCE')])

                                            ]
                                        ),
                                        If(Eq(get_local_id(0), Constant(0)),
                                           [
                                               Assign(ArrayRef(SymbolRef('A'), SymbolRef('groupId')), ArrayRef(SymbolRef('A'), SymbolRef('id')))
                                           ]
                                        )
                                    ]
        ).set_kernel()

        kernel = OclFile("kernel", [apply_one, apply_kernel])

        control = StringTemplate(r"""
        #ifdef __APPLE__
        #include <OpenCL/opencl.h>
        #else
        #include <CL/cl.h>
        #endif
        void apply_all(cl_command_queue queue, cl_kernel kernel, cl_mem buf) {
            size_t global = $n;
            size_t local = 32;
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
            clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);

        }
        """, {'n': Constant(len_A + 32 - (len_A % 32))})        # making it aligned LOL?


        proj = Project([kernel, CFile("generated", [control])])
        fn = ConcreteXorReduction()

        program = cl.clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        apply_kernel_ptr = program['apply_kernel']

        entry_type = ct.CFUNCTYPE(None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem)
        return fn.finalize(apply_kernel_ptr, proj, "apply_all", entry_type)


        # browser_show_ast(tree,'tree_init.png')
        # arg_cfg, tune_cfg = program_config
        # tree = XorReductionFrontend().visit(tree)
        # #browser_show_ast(tree, 'tree_post_frontend.png')
        # tree = XorReductionCBackend(arg_cfg).visit(tree)
        # browser_show_ast(tree, 'tree_post_backend.png')
        # fn = ConcreteXorReduction()
        # arg_type = np.ctypeslib.ndpointer(arg_cfg.dtype, arg_cfg.ndim, arg_cfg.shape, arg_cfg.flags)
        # print(tree.files[0])
        # return fn.finalize('kernel',
        #                    tree,
        #                    ct.CFUNCTYPE(None,arg_type, ct.POINTER(ct.c_int32))
        #                    )

    def points(self, inpt):
        # return np.nditer(inpt)

        # Mihir's Possible Fix... no idea of knowing
        iter = np.nditer(input, flags=['c_index'])
        while not iter.finished:
            yield iter.index
            iter.iternext()



#--------------------------------

class XorOne(LazySlimmy):
    """Xors elements of the array."""

    @staticmethod
    def apply(x, y):
        return x + y


class XorReduction(LazySlimmy):
    def kernel(self, inpt):
        '''
            Calculates the cumulative XOR of elements in inpt, equivalent to
            Reduce with XOR
        '''
        result = 0
        for point in self.points(inpt):
            result = inpt[point] ^ result
        return result


## MAIN EXECUTION ##

def test(apply, arr):
    pass

if __name__ == '__main__':
    # XorReducer = XorReduction()
    # arr = np.array([0b1100,0b1001])
    # print(reduce(lambda x,y: x^y, np.nditer(arr), 0))
    # print(XorReducer(arr))

    #arr = (128*np.random.random(8)).astype(np.int32)
    arr = np.ones(64, np.int32)
    xorer = XorOne()
    output = xorer(arr)
    actual = reduce(lambda x,y : x+y, arr)
    print('output:',[bin(i) for i in output][:64], bin(output[32]))
    print('actual:', actual, bin(actual))
    print('input:',[bin(i) for i in arr][:64])
    print('result:',actual, output[0], actual==output[0])
