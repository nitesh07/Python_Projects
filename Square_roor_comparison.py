import timeit
import time
def sqrt_linear(number):
''' Function will take a number as an argument and return its square root
Args:
number (int) = Number we need to print square root

Returns:
Function will iterate through half till number, and check if the current iterated number multiplied by itself is less than 
the number. Function will return the previous number of the  current iterated number.
''''
    for i in range(number//2):
        if i*i > number:
            return i-1
 
def sqrt_binary_search(number):
  ''' Function will take a number as an argument and return its square root (binary search approach)
Args:
number (int) = Number we need to print square root

Returns:
Function will return mid if mid*mid is equal to number else high.
''''
    low = 1
    high = number//2
    while low<=high:
        mid = (low+high)//2
        if mid*mid == number:
            return mid
        elif mid*mid <number:
            low = mid+1
        else:
            high = mid-1
    return high

__init__ == '__main()__'

start_time = time.time() # start time 
sqrt_linear(1000000000000000)
end_time = time.time() # end time 

linear_time = end_time - start_time
print(f"\nLinear search took {linear_time:.5f} seconds.") 

start_time = time.time() # start time
sqrt_binary_search(1000000000000000)
end_time = time.time() # end time

binary_time = end_time - start_time
print(f"\nLinear search took {binary_time:.5f} seconds.") 
