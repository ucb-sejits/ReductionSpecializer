"""
specializer XorReduction
"""

#
#   INSTRUCTIONS TO RUN THIS CODE
#   Execute this code as follows:
#   >>> python main.py <x> <y> <z>
#   
#   Where <x> is an integer that represents the GPU number. If you don't care, choose 0.
#   Where <y> is an integer that represents the work group size you want. 
#   Where <z> is an integer that represents the size of your dataset of ones. 
#

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




#
### Specialist-Writtern Code ###
#
# The code below is written by an industrt SPECIALIST. This code is meant
# to be more complicated and requires specialized knowledge to write.
#


#
# Global Constants
#

WORK_GROUP_SIZE = 1024
devices = cl.clCreateContextFromType().devices + cl.clCreateContext().devices
TARGET_GPU = devices[1]
ITERATIONS = 0
# print(devices)                # for debugging only

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


class ConcreteReduction(ConcreteSpecializedFunction):
    def __init__(self):
        self.context = cl.clCreateContextFromType()
        self.queue = cl.clCreateCommandQueue(self.context, device=TARGET_GPU)

    def finalize(self, kernel, tree, entry_name, entry_type, compilation_dir=None):
        self.kernel = kernel
        self._c_function = self._compile(entry_name, tree, entry_type, compilation_dir=compilation_dir)
        return self

    def __call__(self, A):
        a = time.time()
        output_array = np.empty(1, A.dtype)
        buf, evt = cl.buffer_from_ndarray(self.queue, A, blocking=False)
        output_buffer, output_evt = cl.buffer_from_ndarray(self.queue, output_array, blocking=False)
        
        b = time.time()
        self._c_function(self.queue, self.kernel, buf, output_buffer)
        c = time.time()
        B, evt = cl.buffer_to_ndarray(self.queue, output_buffer, like=output_array)
        d = time.time()
        # print("overall execution:", d-a, "Initial Copy:", b-a, "Kernel execution:", c-b, "Final Copy:", d-c)
        return B[0]


class LazyUnrolledReduction(LazySpecializedFunction):
    subconfig_type = namedtuple('subconfig', ['dtype', 'ndim', 'shape', 'size', 'flags'])

    def __init__(self, py_ast=None):
        py_ast = py_ast or get_ast(self.apply)
        super(LazyUnrolledReduction, self).__init__(py_ast)


    def args_to_subconfig(self, args):
        A = args[0]  # TODO: currently we only support one argument
        return LazyUnrolledReduction.subconfig_type(A.dtype, A.ndim, A.shape, A.size, [])


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
        fn = ConcreteReduction()

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

class LazyRolledReduction(LazySpecializedFunction):
    subconfig_type = namedtuple('subconfig', ['dtype', 'ndim', 'shape', 'size', 'flags'])

    def __init__(self, py_ast=None):
        py_ast = py_ast or get_ast(self.apply)
        super(LazyRolledReduction, self).__init__(py_ast)

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
        responsible_size = int(len_A / WORK_GROUP_SIZE)         # Get the appropriate number of threads for parallelizing
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

        # Hardcoded OpenCL code to compensate to begin execution of parallelized computation 
        control = StringTemplate(r"""
        #ifdef __APPLE__
        #include <OpenCL/opencl.h>
        #else
        #include <CL/cl.h>
        #endif

        #include <stdio.h>

        #include <sys/time.h>

        double read_timer() {
            struct timeval t;
            struct timezone tz;
            gettimeofday(&t, &tz);
            return (double)t.tv_sec + (double)t.tv_usec * 0.000001;
        }

        void apply_all(cl_command_queue queue, cl_kernel kernel, cl_mem buf, cl_mem out_buf) {
            size_t global = $local;
            size_t local = $local;
            intptr_t len = $length;
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
            clSetKernelArg(kernel, 2, local * sizeof(int), NULL);
            double t0 = read_timer();
            clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            double t1 = read_timer();
            for (int run = 0; run < $runs; run++){
                clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            }
            double t2 = read_timer();
            // printf("Initial Run: %12.9f \n", t1 - t0);
            if ($runs){
                printf("Average time: %12.9f \n", (t2 - t1) / $runs);
            }
        }
        """, {'local': Constant(WORK_GROUP_SIZE),
              'n': Constant((len_A + WORK_GROUP_SIZE - (len_A % WORK_GROUP_SIZE))/2),
              'length': Constant(len_A),
              'runs': Constant(ITERATIONS)
        })

        proj = Project([kernel, CFile("generated", [control])])
        fn = ConcreteReduction()

        program = cl.clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        apply_kernel_ptr = program['apply_kernel']
        entry_type = ct.CFUNCTYPE(None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem)
        return fn.finalize(apply_kernel_ptr, proj, "apply_all", entry_type, compilation_dir=self.config_to_dirname(program_config))

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
        fn = ConcreteReduction()

        program = cl.clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        apply_kernel_ptr = program['apply_kernel']
        entry_type = ct.CFUNCTYPE(None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem)
        return fn.finalize(apply_kernel_ptr, proj, "apply_all", entry_type)


#
### User-Writtern Code ###
#
# The code below is written by a USER. This code is meant to be 
# simple and easy to write
#

class Baseline(CopyBaseline):
    @staticmethod
    def apply(x, y):
        return x + y

#
# Xor Reduction
#
class UnRolledXor(LazyUnrolledReduction):
    """
       Xors elements of an input array. Inherits from LazyUnrolledReduction, 
       so the loop is unrolled. An example is given for how to use this class
       by calling it on a particular array.

       >>> arr = [7, 8, 5, 6, 3, 1, 5, 2]
       >>> unrolled = UnRolledXor()
       >>> unrolled(arr)                # returns the xor of all elements of arr

    """
    @staticmethod
    def apply(x, y):
        return x ^ y


class RolledXor(LazyRolledReduction):
    """
       Xors elements of an input array. Inherits from LazyRolledReduction, 
       so the loop is not unrolled. An example is given for how to use this class
       by calling it on a particular array.

       >>> arr = [7, 8, 5, 6, 3, 1, 5, 2]
       >>> rolled = RolledXor()
       >>> rolled(arr)                  # returns the xor of all elements of arr

    """
    @staticmethod
    def apply(x, y):
        return x ^ y

#
# Addition Reduction
#
class UnRolledAdd(LazyUnrolledReduction):
    """
       Adds the elements of an input array. Inherits from LazyUnrolledReduction, 
       so the loop is unrolled. An example is given for how to use this class
       by calling it on a particular array.

       >>> arr = [7, 8, 5, 6, 3, 1, 5, 2]
       >>> unrolled = UnRolledAdd()
       >>> unrolled(arr)                  # returns the sum of all elements of arr

    """
    @staticmethod
    def apply(x, y):
        return x + y


class RolledAdd(LazyRolledReduction):
    """
       Adds the elements of an input array. Inherits from LazyRolledReduction, 
       so the loop is not unrolled. An example is given for how to use this class
       by calling it on a particular array.

       >>> arr = [7, 8, 5, 6, 3, 1, 5, 2]
       >>> rolled = RolledAdd()
       >>> rolled(arr)                  # returns the sum of all elements of arr

    """
    @staticmethod
    def apply(x, y):
        return x + y



#
### Main Execution (currently used for testing) ###
#

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
    device_num = int(sys.argv[1])                                           # gets the device number from command line args
    TARGET_GPU = devices[device_num]

    WORK_GROUP_SIZE = int(sys.argv[2]) or TARGET_GPU.max_work_group_size
    size = int(eval(sys.argv[3]))
    print(TARGET_GPU, WORK_GROUP_SIZE)

    arr = (np.ones(size)*8).astype(np.float32)                              # used for creation of a dataset with all 1's

    baseline = Baseline()                                               
    rolled = RolledAdd()

    res = rolled(arr)
    npres = np.sum(arr)
    print('Rolled Code: ', res, type(res), 'Numpy: ', npres, type(npres), abs(npres - res) < 1e-8)





