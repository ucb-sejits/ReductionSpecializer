#
# Importations
#

from __future__ import print_function, division
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.frontend import get_ast
from ctree import browser_show_ast
from ctree.transformations import PyBasicConversions
from ctree.nodes import CtreeNode, Project
from ctree.c.nodes import For, SymbolRef, Assign, Lt, PostInc, \
    Constant, Deref, FunctionDecl, Add, Mul, If, And, ArrayRef, FunctionCall, \
    CFile, Eq, Mod, AugAssign, MulAssign, LtE, String, Div, Gt, BitShRAssign, BitAnd, Sub
import ctree.np
import ctypes as ct
from ctree.ocl.macros import get_global_id, get_group_id, get_local_id, barrier, CLK_LOCAL_MEM_FENCE
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from math import log, ceil
from collections import namedtuple

import numpy as np
import pycl as cl
import sys, time
import ast
import logging


class ReductionCBackend(ast.NodeTransformer):
    def __init__(self, arg_cfg):
        self.arg_cfg = arg_cfg
        self.retval = None

    def visit_FunctionDecl(self, node):
        arg_type = np.ctypeslib.ndpointer(self.arg_cfg.dtype, self.arg_cfg.ndim,
                                          self.arg_cfg.shape)  # arg_type is the c-type of the input (like int *)

        # TODO: stripping out self, as is done below, should really be abstracted, as it's the same everywhere
        # Get the actual params (stripping out "self" from node.params)
        param = node.params[1]          # note that params[0] is self
        param.type = arg_type()
        node.params = [param]           # this basically strips out "self"

        # # Alternative to above code; stripping out "self" from node.params
        # node.params[1].type = argtype()
        # node.params = node.params[1:]

        # TODO: below, 'output' should really be (int *) hardcoded, rather than the same
        #       as the input type (which is represented by argtype)

        ## Adding the 'output' variable as one of the parameters of type argtype
        retval = SymbolRef("output", arg_type())  # retval is a symbol reference to c-variable named "output" of type argtype
        self.retval = "output"                    # 'output' is the name of the value to return
        node.params.append(retval)                # this appends the output parameter to the list of parameters
        node.defn = list(map(self.visit, node.defn))
        node.defn[0].left.type = arg_type._dtype_.type()
        return node

    def visit_PointsLoop(self, node):
        target = node.target
        return For(
            Assign(SymbolRef(target, ct.c_int()), Constant(0)),  # int i = 0;
            Lt(SymbolRef(target), Constant(self.arg_cfg.size)),  # 'Lt' = Less than; i < size of array
            PostInc(SymbolRef(target)),                          # i++
            list(map(self.visit, node.body))                     # Recursively call the other nodes
        )

    def visit_Return(self, node):
        return Assign(Deref(SymbolRef(self.retval)), node.value)  # *output = <return value>




class CopyBaseline(LazySpecializedFunction):
    subconfig_type = namedtuple('subconfig', ['dtype', 'ndim', 'shape', 'size', 'flags'])


    def args_to_subconfig(self, args):
        # what happens if you have more than one arg?
        A = args[0]
        return self.subconfig_type(A.dtype, A.ndim, A.shape, A.size, [])


    def transform(self, tree, program_config):
        A = program_config[0]
        len_A = np.prod(A.shape)
        inner_type = A.dtype.type()
        pointer = np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape)
        apply_one = PyBasicConversions().visit(tree.body[0])
        apply_one.return_type = inner_type
        apply_one.params[0].type = inner_type
        apply_one.params[1].type = inner_type
        responsible_size = int(len_A / WORK_GROUP_SIZE)
        apply_kernel = FunctionDecl(None, "apply_kernel",
                                    params=[SymbolRef("A", pointer()).set_global(),
                                            SymbolRef("output_buf", pointer()).set_global(),
                                            SymbolRef("localData", pointer()).set_local()
                                    ],
                                    defn=[
                                        Constant(1)
                                    ]
        ).set_kernel()

        kernel = OclFile("kernel", [apply_one, apply_kernel])

        control = StringTemplate(r"""
        #ifdef __APPLE__
        #include <OpenCL/opencl.h>
        #else
        #include <CL/cl.h>
        #endif

        #include <stdio.h>

        void apply_all(cl_command_queue queue, cl_kernel kernel, cl_mem buf, cl_mem out_buf) {
            size_t global = 1;
            size_t local = 1;
            intptr_t len = 1;
            cl_mem swap;
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
            clSetKernelArg(kernel, 2, local * sizeof(int), NULL);
            clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        }
        """, {})

        proj = Project([kernel, CFile("generated", [control])])
        fn = ConcreteReduction()

        program = cl.clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        apply_kernel_ptr = program['apply_kernel']
        entry_type = ct.CFUNCTYPE(None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem)
        return fn.finalize(apply_kernel_ptr, proj, "apply_all", entry_type)


### Frontend ###

class PointsLoop(CtreeNode):
    _fields = ['target', 'iter_target', 'body']

    def __init__(self, target, iter_target, body):
        self.target = target
        self.iter_target = iter_target
        self.body = body
        # super(PointsLoop, self).__init__()

    def label(self):
        return str(self.iter_target)


class ReductionFrontend(PyBasicConversions):
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
