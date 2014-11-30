import main
import numpy as np


CORRECTNESS_THRESHOLD = 1e-8

num_tests = 0			# counter for the total number of tests run
num_correct	= 0			# counter for the total number of tests passed


def run_tests():
	arr = (np.ones(2^0) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^1) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	# FIXME: This tests fails
	arr = (np.ones(2^2) * 8).astype(np.float32)
	# check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^3) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^4) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^5) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^6) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^7) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^8) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^9) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^10) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^11) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^12) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^13) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^14) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^15) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^16) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^17) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^19) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^20) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^21) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^22) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^23) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^24) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^25) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^26) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^27) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^28) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^29) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

	arr = (np.ones(2^30) * 8).astype(np.float32)
	check_test(do_test(arr), do_control(arr))

def do_test(arr, work_group_size=None, target_gpu_index=0):

	main.TARGET_GPU = main.devices[target_gpu_index]
	main.WORK_GROUP_SIZE = work_group_size or main.TARGET_GPU.max_work_group_size

	rolled = main.Rolled()
	return rolled(arr)


def do_control(arr):
	return np.sum(arr)


def check_test(test_value, correct_value):
	global num_tests
	global num_correct

	num_tests += 1
	if abs(test_value - correct_value) < CORRECTNESS_THRESHOLD:
		num_correct += 1
		print "Passed Test " + str(num_tests)
		return True
	else:
		print "Test " + str(num_tests) + " FAILED: Expected " + str(correct_value) + " but got " + str(test_value) + "."
		return False

def print_final_status():
	if num_tests == num_correct:
		print "ALL " + str(num_tests) + " TESTS PASSED"
	else:
		print "PASSED " + str(num_correct) + " OF " + str(num_tests) + " TESTS"

run_tests()
print_final_status()