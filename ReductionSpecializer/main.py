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
from ctree.frontend import get_ast, dump
from ctree import browser_show_ast
from ctree.transformations import PyBasicConversions
from ctree.nodes import CtreeNode, Project
from ctree.types import get_c_type_from_numpy_dtype
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


class ConcreteReduction(ConcreteSpecializedFunction):
    def __init__(self):
        self.context = cl.clCreateContextFromType()
        self.queue = cl.clCreateCommandQueue(self.context, device=TARGET_GPU)

    def finalize(self, kernel, tree, entry_name, entry_type):
        self.kernel = kernel
        self._c_function = self._compile(entry_name, tree, entry_type)
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


    def args_to_subconfig(self, args):
        A = args[0]  # TODO: currently we only support one argument
        return LazyUnrolledReduction.subconfig_type(A.dtype, A.ndim, A.shape, A.size, [])


    def transform(self, tree, program_config):
        A = program_config[0]
        len_A = np.prod(A.shape)
        inner_type = get_ctype(A.dtype)()
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

        c_controller = CFile("generated", [control])
        return [kernel, c_controller]

    def finalize(self, transform_result, program_config):
        ocl_kernel = transform_result[0]
        c_controller = transform_result[1]
        proj = Project([ocl_kernel, c_controller])
        fn = ConcreteReduction()                        # define the ConcreteSpecializeFunction subclass to use

        program = cl.clCreateProgramWithSource(fn.context, ocl_kernel.codegen()).build()
        apply_kernel_ptr = program['apply_kernel']
        entry_type = ct.CFUNCTYPE(None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem)
        return fn.finalize(apply_kernel_ptr, proj, "apply_all", entry_type)

class LazyRolledReduction(LazySpecializedFunction):
    subconfig_type = namedtuple('subconfig', ['dtype', 'ndim', 'shape', 'size', 'flags'])

    def args_to_subconfig(self, args):
        # what happens if you have more than one arg?
        A = args[0]
        return self.subconfig_type(A.dtype, A.ndim, A.shape, A.size, [])


    def transform(self, tree, program_config):
        dirname = self.config_to_dirname(program_config)
        A = program_config[0]
        len_A = np.prod(A.shape)
        data_type = get_c_type_from_numpy_dtype(A.dtype)        # Get the ctype class for the data type for the parameters
        pointer = np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape)
        apply_one = PyBasicConversions().visit(tree.body[0])      
          
        apply_one.name = 'apply'                                # Naming our kernel method

        # Assigning a data_type instance for the  #
        # return type, and the parameter types... #
        apply_one.return_type = data_type()                     
        apply_one.params[0].type = data_type()
        apply_one.params[1].type = data_type()

        responsible_size = int(len_A / WORK_GROUP_SIZE)         # Get the appropriate number of threads for parallelizing
        
        # Creating our controller function (called "apply_kernel") to control #
        # the parallelizing of our computation, using ctree syntax...         #
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
                                                       FunctionCall(apply_one.name, [SymbolRef('localResult'),
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
                                                Assign(SymbolRef('localResult'), FunctionCall(SymbolRef(apply_one.name), [SymbolRef('localResult'),
                                                                                                                   ArrayRef(SymbolRef('localData'),Constant(x))]))
                                                for x in range(1, WORK_GROUP_SIZE)
                                           ] + [Assign(ArrayRef(SymbolRef('output_buf'), Constant(0)), SymbolRef('localResult'))]
                                        )
                                    ]
        ).set_kernel()

        # Hardcoded OpenCL code to compensate to begin execution of parallelized computation 
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
              'length': Constant(len_A),
        })

        ocl_kernel = OclFile("kernel", [apply_one, apply_kernel])
        c_controller = CFile("generated", [control])
        return [ocl_kernel, c_controller]

    def finalize(self, transform_result, program_config):
        ocl_kernel = transform_result[0]
        c_controller = transform_result[1]
        proj = Project([ocl_kernel, c_controller])
        fn = ConcreteReduction()                        # define the ConcreteSpecializeFunction subclass to use

        program = cl.clCreateProgramWithSource(fn.context, ocl_kernel.codegen()).build()
        apply_kernel_ptr = program['apply_kernel']
        entry_type = ct.CFUNCTYPE(None, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem)
        return fn.finalize(apply_kernel_ptr, proj, "apply_all", entry_type)

#
### User-Written Code ###
#
# The code below is written by a USER. This code is meant to be 
# simple and easy to write
#

def add(x, y):
    """
       Kernel method to add up all the elements in an array. An example is given in the
       doctests.

       >>> RolledClass = LazyRolledClassReduction.from_function(add, "RolledClass")
       >>> rolled = RolledClass()
       >>> rolled(arr)                     # returns the sum of all elements of arr
    """
    # x, y = x + y, x - y
    # adam = 5
    return x + y


if __name__ == '__main__':
    ## Setup to use command-line arguments ##
    device_num = int(sys.argv[1])                                               # gets the device number from command line args
    TARGET_GPU = devices[device_num]

    WORK_GROUP_SIZE = int(sys.argv[2]) or TARGET_GPU.max_work_group_size
    size = int(eval(sys.argv[3]))

    ## Sample dataset creation ##
    sample_data = (np.ones(size)*8).astype(np.float32)                          # creating a dataset with all 8's
      
    ##################################################################################                                       
    ## EXAMPLE 1: Rolled Reduction Example (using the add() function defined above) ##
    ##################################################################################
    RolledClass = LazyRolledReduction.from_function(add, "RolledClass")         # generate subclass with the add(x, y) method, defined above
    reducer_conventional = RolledClass()                                        # create your reducer             
    sejits_result_conventional = reducer_conventional(sample_data)              # the result of the SEJITS reduction


    ###################################################################                                       
    ## EXAMPLE 2: Rolled Reduction Example (using a lambda function) ##
    ###################################################################                                       
    sum_kernel = lambda x, y: x + y                                                         # create your lambda function
    RolledClassLambda = LazyRolledReduction.from_function(sum_kernel, "RolledClassLambda")  # generate subclass with the sum_kernel() lambda function we just defined
    reducer_lambda = RolledClassLambda()                                                    # create your reducer             
    sejits_result_lambda = reducer_lambda(sample_data)                                      # the result of the SEJITS reduction



    ## Running the control (using numpy) for testing ##
    numpy_result = np.add.reduce(sample_data)

    ## Printing out the result ##
    print('SEJITS RESULT (Lambda): \t', sejits_result_lambda, " of ", type(sejits_result_lambda))
    print('SEJITS RESULT (Conventional): \t', sejits_result_conventional, " of ", type(sejits_result_conventional))
    print ('NUMPY RESULT: \t\t\t', numpy_result, " of ", type(numpy_result))
    print ('SUCCESS?: \t\t\t', abs(numpy_result - sejits_result_lambda) < 1e-8 and abs(numpy_result - sejits_result_conventional) < 1e-8)





