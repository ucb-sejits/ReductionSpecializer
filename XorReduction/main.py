"""
specializer XorReduction
"""

from __future__ import print_function, division

import logging

#logging.basicConfig(level=20)

import numpy as np
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
from ctree.ocl.macros import get_global_id, get_group_id, get_local_id, get_global_size, barrier, CLK_LOCAL_MEM_FENCE
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
import pycl as cl



from math import log, ceil
import sys, time

from math import ceil

WORK_GROUP_SIZE = 1024
devices = cl.clCreateContextFromType().devices + cl.clCreateContext().devices
print(devices)
TARGET_GPU = devices[1]

import ast

from collections import namedtuple


class PointsLoop(CtreeNode):
    _fields = ['target', 'iter_target', 'body']

    def __init__(self, target, iter_target, body):
        self.target = target
        self.iter_target = iter_target
        self.body = body
        # super(PointsLoop, self).__init__()

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
        arg_type = np.ctypeslib.ndpointer(self.arg_cfg.dtype, self.arg_cfg.ndim,
                                          self.arg_cfg.shape)  # arg_type is the c-type of the input (like int *)

        # TODO: stripping out self, as is done above, should really be abstracted, as it's the same everywhere
        # Get the actual params (stripping out "self" from node.params)
        param = node.params[1]  # note that params[0] is self
        param.type = arg_type()  # because logic
        node.params = [param]  # this basically strips out "self"

        # # Alternative to above code; stripping out "self" from node.params
        # node.params[1].type = argtype()
        # node.params = node.params[1:]

        # TODO: below, 'output' should really be (int *) hardcoded, rather than the same
        #       as the input type (which is represented by argtype)
        ## Adding the 'output' variable as one of the parameters of type argtype
        retval = SymbolRef("output",
                           arg_type())  # retval is a symbol reference to c-variable named "output" of type argtype
        self.retval = "output"  # 'output' is the name of
        node.params.append(retval)  # this appends the output parameter to the list of parameters
        node.defn = list(map(self.visit, node.defn))
        node.defn[0].left.type = arg_type._dtype_.type()
        return node

    def visit_PointsLoop(self, node):
        target = node.target
        return For(
            # TODO: Not sustainable... what happens i starts at 1?
            Assign(SymbolRef(target, ct.c_int()), Constant(0)),  # int i = 0;
            Lt(SymbolRef(target), Constant(self.arg_cfg.size)),  # 'Lt' = Less than; i < size of array
            PostInc(SymbolRef(target)),  # i++
            list(map(self.visit, node.body))  # Recursively call the other nodes
        )

    def visit_Return(self, node):
        return Assign(Deref(SymbolRef(self.retval)), node.value)  # *output = <return value>


class ConcreteXorReduction(ConcreteSpecializedFunction):
    def __init__(self):
        self.context = cl.clCreateContextFromType()
        self.queue = cl.clCreateCommandQueue(self.context, device=TARGET_GPU)

    def finalize(self, kernel, tree, entry_name, entry_type):
        self.kernel = kernel
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, A):
        output_array = np.empty(1, A.dtype)
        buf, evt = cl.buffer_from_ndarray(self.queue, A, blocking=False)
        output_buffer, output_evt = cl.buffer_from_ndarray(self.queue, output_array, blocking=False)
        self._c_function(self.queue, self.kernel, buf, output_buffer)
        B, evt = cl.buffer_to_ndarray(self.queue, output_buffer, like=output_array)
        return B[0]


class LazySlimmy(LazySpecializedFunction):
    subconfig_type = namedtuple('subconfig', ['dtype', 'ndim', 'shape', 'size', 'flags'])

    def __init__(self, py_ast=None):
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
        responsible_size = int(len_A / WORK_GROUP_SIZE)
        apply_kernel = FunctionDecl(None, "apply_kernel",
                                    params=[SymbolRef("A", pointer()).set_global(),
                                            SymbolRef("output_buf", pointer()).set_global(),
                                            SymbolRef("localData", pointer()).set_local()
                                    ],
                                    defn=[
                                        Assign(SymbolRef('groupId', ct.c_int()), get_group_id(0)),
                                        Assign(SymbolRef('globalId', ct.c_int()), get_global_id(0)),
                                        Assign(SymbolRef('localId', ct.c_int()), get_local_id(0)),
                                        Assign(SymbolRef('localResult', ct.c_int()),
                                               ArrayRef(SymbolRef('A'), SymbolRef('globalId'))
                                               )
                                        ] +
                                        [Assign(SymbolRef('localResult'),
                                                FunctionCall(SymbolRef('apply'),
                                                             [SymbolRef('localResult'), ArrayRef(SymbolRef('A'),Add(SymbolRef('globalId'), Constant(i * WORK_GROUP_SIZE)))]))
                                            for i in range(1, responsible_size)] +
                                        [
                                            Assign(ArrayRef(SymbolRef('localData'), SymbolRef('globalId')),
                                                SymbolRef('localResult')
                                               ),
                                            barrier(CLK_LOCAL_MEM_FENCE()),
                                        If(Eq(SymbolRef('globalId'), Constant(0)),
                                           [
                                                Assign(SymbolRef('localResult'), FunctionCall(SymbolRef('apply'), [SymbolRef('localResult'),
                                                                                                                   ArrayRef(SymbolRef('localData'),Constant(x))]))
                                                for x in range(1, WORK_GROUP_SIZE)
                                           ] + [Assign(ArrayRef(SymbolRef('output_buf'), Constant(0)), SymbolRef('localResult'))]
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

        #include <stdio.h>

        void apply_all(cl_command_queue queue, cl_kernel kernel, cl_mem buf, cl_mem out_buf) {
            size_t global = $local;
            size_t local = $local;
            intptr_t len = $length;
            cl_mem swap;
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
            clSetKernelArg(kernel, 2, local * sizeof(int), NULL);
            clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        }
        """, {'local': Constant(WORK_GROUP_SIZE),
              'n': Constant((len_A + WORK_GROUP_SIZE - (len_A % WORK_GROUP_SIZE))/2),
              'length': Constant(len_A)
        })

        proj = Project([kernel, CFile("generated", [control])])
        fn = ConcreteXorReduction()

        program = cl.clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        apply_kernel_ptr = program['apply_kernel']
        entry_type = ct.CFUNCTYPE(None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem)
        return fn.finalize(apply_kernel_ptr, proj, "apply_all", entry_type)

    def points(self, inpt):
        # return np.nditer(inpt)

        iter = np.nditer(input, flags=['c_index'])
        while not iter.finished:
            yield iter.index
            iter.iternext()

class RolledSlimmy(LazySpecializedFunction):
    subconfig_type = namedtuple('subconfig', ['dtype', 'ndim', 'shape', 'size', 'flags'])

    def __init__(self, py_ast=None):
        py_ast = py_ast or get_ast(self.apply)
        super(RolledSlimmy, self).__init__(py_ast)

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
                                        Assign(SymbolRef('groupId', ct.c_int()), get_group_id(0)),
                                        Assign(SymbolRef('globalId', ct.c_int()), get_global_id(0)),
                                        Assign(SymbolRef('localId', ct.c_int()), get_local_id(0)),
                                        Assign(SymbolRef('localResult', (ct.c_int() if A.dtype is np.int32 else ct.c_float())),
                                               ArrayRef(SymbolRef('A'), SymbolRef('globalId'))
                                               ),
                                        For(Assign(SymbolRef('offset', ct.c_int()), Constant(1)), Lt(SymbolRef('offset'), Constant(responsible_size)),
                                            PostInc(SymbolRef('offset')),
                                            [
                                                Assign(SymbolRef('localResult'),
                                                       FunctionCall('apply', [SymbolRef('localResult'),
                                                                              ArrayRef(SymbolRef('A'),
                                                                                       Add(SymbolRef('globalId'),
                                                                                           Mul(SymbolRef('offset'),
                                                                                               Constant(WORK_GROUP_SIZE))))])
                                                       ),
                                            ]
                                        ),
                                            Assign(ArrayRef(SymbolRef('localData'), SymbolRef('globalId')),
                                                SymbolRef('localResult')
                                               ),
                                            barrier(CLK_LOCAL_MEM_FENCE()),
                                        If(Eq(SymbolRef('globalId'), Constant(0)),
                                           [
                                                Assign(SymbolRef('localResult'), FunctionCall(SymbolRef('apply'), [SymbolRef('localResult'),
                                                                                                                   ArrayRef(SymbolRef('localData'),Constant(x))]))
                                                for x in range(1, WORK_GROUP_SIZE)
                                           ] + [Assign(ArrayRef(SymbolRef('output_buf'), Constant(0)), SymbolRef('localResult'))]
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

        #include <stdio.h>

        void apply_all(cl_command_queue queue, cl_kernel kernel, cl_mem buf, cl_mem out_buf) {
            size_t global = $local;
            size_t local = $local;
            intptr_t len = $length;
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
            clSetKernelArg(kernel, 2, local * sizeof(int), NULL);
            clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        }
        """, {'local': Constant(WORK_GROUP_SIZE),
              'n': Constant((len_A + WORK_GROUP_SIZE - (len_A % WORK_GROUP_SIZE))/2),
              'length': Constant(len_A)
        })

        proj = Project([kernel, CFile("generated", [control])])
        fn = ConcreteXorReduction()

        program = cl.clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        apply_kernel_ptr = program['apply_kernel']
        entry_type = ct.CFUNCTYPE(None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem)
        return fn.finalize(apply_kernel_ptr, proj, "apply_all", entry_type)

class CopyBaseline(LazySpecializedFunction):
    subconfig_type = namedtuple('subconfig', ['dtype', 'ndim', 'shape', 'size', 'flags'])

    def __init__(self, py_ast=None):
        py_ast = py_ast or get_ast(self.apply)
        super(CopyBaseline, self).__init__(py_ast)

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
        fn = ConcreteXorReduction()

        program = cl.clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        apply_kernel_ptr = program['apply_kernel']
        entry_type = ct.CFUNCTYPE(None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem)
        return fn.finalize(apply_kernel_ptr, proj, "apply_all", entry_type)





# --------------------------------

class Baseline(CopyBaseline):
    @staticmethod
    def apply(x, y):
        return x + y

class XorOne(LazySlimmy):
    """Xors elements of the array."""

    @staticmethod
    def apply(x, y):
        return x ^ y

class UnRolled(LazySlimmy):

    @staticmethod
    def apply(x, y):
        return x+y

class RolledXor(CopyBaseline):
    @staticmethod
    def apply(x, y):
        return x^y

class Rolled(RolledSlimmy):
    @staticmethod
    def apply(x, y):
        #return 0*x + 0*y - -1*x - -1*y+1*x + 1*y - 0*x - 0*y+2*x + 2*y - 1*x - 1*y+3*x + 3*y - 2*x - 2*y+4*x + 4*y - 3*x - 3*y+5*x + 5*y - 4*x - 4*y+6*x + 6*y - 5*x - 5*y+7*x + 7*y - 6*x - 6*y+8*x + 8*y - 7*x - 7*y+9*x + 9*y - 8*x - 8*y+10*x + 10*y - 9*x - 9*y+11*x + 11*y - 10*x - 10*y+12*x + 12*y - 11*x - 11*y+13*x + 13*y - 12*x - 12*y+14*x + 14*y - 13*x - 13*y+15*x + 15*y - 14*x - 14*y+16*x + 16*y - 15*x - 15*y+17*x + 17*y - 16*x - 16*y+18*x + 18*y - 17*x - 17*y+19*x + 19*y - 18*x - 18*y
        return x+y

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

def timeit(f, args):
    a = time.time()
    f(*args)
    return time.time() - a

def average_time(f, args, iterations):
    a = time.time()
    i = iterations
    while i:
        f(*args)
        i -= 1
    return (time.time() - a) / iterations

def interleaved_timing(fs, args, iterations):
    times = {f:[] for f in fs}
    i = iterations
    while i:
        for f in fs:
            a = time.time()
            f(*args)
            times[f].append(time.time() - a)
        i -= 1
    return {f: sum(t)/iterations for f,t in times.items()}

if __name__ == '__main__':
    device_num = int(sys.argv[1])
    TARGET_GPU = devices[device_num]
    WORK_GROUP_SIZE = int(sys.argv[2]) or TARGET_GPU.max_work_group_size
    print(TARGET_GPU, WORK_GROUP_SIZE)
    arr = (np.random.rand(int(eval(sys.argv[3])))*8).astype(np.float32)
    baseline = Baseline()
    #baseline = lambda x : None
    rolled = Rolled()
    #unrolled = UnRolled()

    times = interleaved_timing((baseline, rolled, np.sum), [arr], 16)
    rolled_time = times[rolled]
    baseline_time = times[baseline]
    np_time = times[np.sum]
    print('Rolled:', rolled_time, 'Baseline:', baseline_time,'NP:', np_time, 'Delta:', rolled_time - baseline_time)
    print('Rolled', rolled(arr), 'NP:', np.sum(arr))


