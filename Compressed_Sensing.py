import random

import numpy
from numpy.random import RandomState
from scipy import sparse

'''
Compression theory y = Ax
y is resultant measurements (K x 1)
A is gaussian distributed matrix(K x N)
x is input measurement(N x 1)
'''
numpy.set_printoptions(precision=3)
numpy.set_printoptions(linewidth=600)

'''
Generation of sparse matrix x(N x 1)
Matrix has one row and random number(20 < n < 50) columns
Represents input data x
'''

n = random.randint(20, 50)
# Density is best if between 0.1 and 0.25
density = float(input("\n\n\nEnter density of sparse input matrix.Enter 0 for random density\n"))
if density == 0:
    density = random.uniform(0.1, 0.25)
x = sparse.random(n, 1, density, random_state=RandomState(1))
print("\n\n\nGenerating input matrix with ", n, " values and ", density, "density")
x = x.todense()
x.reshape(n, 1)
print(x)

# Generation of Gaussian matrix A(K x N)
k = int(input("\n\nEnter value of k\n"))
A = numpy.random.normal(size=(k*n))
print("Generating gaussian matrix of order ", k, "x", n)
A = A.reshape(k, n)
print(A)
print("\n")
#  Multiplication of Matrices
y = numpy.dot(A, x)
print("\n\nResultant matrix y from Ax of order ", k, "x 1\n\n")
print(y)
