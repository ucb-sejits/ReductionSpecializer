import main
import time
import numpy as np



CORRECTNESS_THRESHOLD = 1e-8

num_tests = 0			# counter for the total number of tests run
num_correct	= 0			# counter for the total number of tests passed


def run_tests():

	for i in range(28):
		arr = (np.ones(2**i) * 8).astype(np.float32)
		test_val, test_time = do_test(arr)
		correct_val, control_time = do_control(arr, np.sum)
		check_test(test_val, correct_val, test_time, control_time)



def do_test(arr, work_group_size=None, target_gpu_index=0):

	try:
		main.TARGET_GPU = main.devices[target_gpu_index]
		main.WORK_GROUP_SIZE = work_group_size or main.TARGET_GPU.max_work_group_size

		rolled = main.RolledAdd()

		start = time.time()
		result =  rolled(arr)
		finish = time.time()

		return result, finish - start
	except Exception as e:
		return e, time.time() - start

def do_control(arr, func):

	start = time.time()
	control_result = func(arr)
	finish = time.time()

	return control_result, finish - start


def check_test(test_value, correct_value, test_time, control_time):
	global num_tests
	global num_correct

	num_tests += 1

	test_time_string = "\t [ SEJITS: {0:.10} sec".format(test_time)
	control_time_string = "\t|     NUMPY: {0:.10} sec ]".format(control_time)
	time_string = test_time_string + control_time_string

	if isinstance(test_value, Exception):
		print bcolors.FAIL + "\t" + u'\u2717'  + "   Test " + str(num_tests) + " FAILED" + bcolors.GRAY + "\tThrew exception: " + repr(test_value) + time_string +  bcolors.ENDC	
	elif abs(test_value - correct_value) < CORRECTNESS_THRESHOLD:
		num_correct += 1
		print bcolors.GREEN + "\t" + u'\u2713' + bcolors.GRAY + "   Test " + str(num_tests) + time_string + bcolors.ENDC
	else:
		print bcolors.FAIL + "\t" + u'\u2717'  + "   Test " + str(num_tests) + " FAILED" + bcolors.GRAY + "\tExpected " + str(correct_value) + " but got " + str(test_value) +  time_string + bcolors.ENDC

def print_final_status():
	if num_tests == num_correct:
		print bcolors.GREEN + "\tALL " + str(num_tests) + " TESTS PASSED" + bcolors.ENDC
	else:
		print bcolors.FAIL + "\tPASSED " + str(num_correct) + " OF " + str(num_tests) + " TESTS" + bcolors.ENDC


class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    GRAY = '\033[90m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


run_tests()
print_final_status()