''' ==========================================
== Written by..: Manuel Puello            ====
== Date Written: April 01                 ====
== Purpose.....: Assignment 3             ====
==============================================
'''
import numpy as np

# 1 - Euler Method with the following details:
# a. Function: t – y^2
# b. Range: 0 < t < 2
# c. Iterations: 10
# d. Initial Point: f(0) = 1
def parametric_function(t, y):
    return (t - y ** 2)

def euler_method(t, y, iterations, x):
    h = (x - t) / iterations

    for available_variable in range(iterations):
        y = y + (h * parametric_function(t, y))
        t = t + h

    print("%.5f" % y)
    print("\n")

# 2 - Runge-Kutta with the following details:
# a. Function: t – y^2
# b. Range: 0 < t < 2
# c. Iterations: 10
# d. Initial Point: f(0) = 1
def parametric_function_2(t, y):
    return (t - y ** 2)

def runge_kutta(t, y, iterations, x):
    h = (x - t) / iterations

    for available_variable_2 in range(iterations):
        k_1 = h * parametric_function_2(t, y)
        k_2 = h * parametric_function_2((t + (h / 2)), (y + (k_1 / 2)))
        k_3 = h * parametric_function_2((t + (h / 2)), (y + (k_2 / 2)))
        k_4 = h * parametric_function_2((t + h), (y + k_3))

        y = y + (1 / 6) * (k_1 + (2 * k_2) + (2 * k_3) + k_4)

        t = t + h

    print("%.5f" % y)
    print("\n")

# 3 - Use Gaussian elimination and backward substitution solve the following linear system of
# equations written in augmented matrix format.
# [2 -1 1 | 6
#  1  3 1 | 0
# -1  5 4 |-3]
def gauss_elimination(matrix):
    size = matrix.shape[0]

    for i in range(size):
        pivot = i
        while matrix[pivot, i] == 0:
            pivot += 1

        matrix[[i, pivot]] = matrix[[pivot, i]]

        for j in range(i + 1, size):
            factor = matrix[j, i] / matrix[i, i]
            matrix[j, i:] = matrix[j, i:] - factor * matrix[i, i:]

    inputs = np.zeros(size)

    for i in range(size - 1, -1, -1):
        inputs[i] = (matrix[i, -1] - np.dot(matrix[i, i: -1], inputs[i:])) / matrix[i, i]

    output = np.array([int(inputs[0]), int(inputs[1]), int(inputs[2])], dtype=np.double)
    print(output)
    print("\n")

# 4 - Implement LU Factorization for the following matrix and do the following:
#[ 1  1  0  3
#  2  1 -1  1
#  3 -1 -1  2
# -1  2  3 -1]
# a. Print out the matrix determinant.
# b. Print out the L matrix.
# c. Print out the U matrix.
def LU_factorization(matrix_2):
    size = matrix_2.shape[0]

    l_factor = np.eye(size)
    u_factor = np.zeros_like(matrix_2)

    for i in range(size):
        for j in range(i, size):
            u_factor[i, j] = (matrix_2[i, j] - np.dot(l_factor[i, :i], u_factor[:i, j]))

        for j in range(i + 1, size):
            l_factor[j, i] = (matrix_2[j, i] - np.dot(l_factor[j, :i], u_factor[:i, i])) / u_factor[i, i]

    determinant = np.linalg.det(matrix_2)

    print("%.5f" % determinant)
    print("\n")
    print(l_factor)
    print("\n")
    print(u_factor)
    print("\n")

#5 - Determine if the following matrix is diagonally dominate
#[  9  0  5  2  1
#   3  9  1  2  1
#   0  1  7  2  3
#   4  3  2  12 2
#   3  2  4  0  8]
def diagonally_dominate(dominate_matrix, n):
    for i in range(0, n):
        total = 0
        for j in range(0, n):
            total = total + abs(dominate_matrix[i][j])

        total = total - abs(dominate_matrix[i][i])

    if abs(dominate_matrix[i][i]) < total:
        print("False\n")
    else:
        print("True\n")

    print("\n")


## 6 - Determine if the matrix is a positive definite
#[  2  2  1
#   2  3  0
#   1  0  2]
def positive_definite(definite_matrix):
    eigenvalues = np.linalg.eigvals(definite_matrix)

    if np.all(eigenvalues > 0):
        print("True\n")
    else:
        print("False\n")

    print("\n")

if __name__ == "__main__":
    # Euler Method Function
    t_0 = 0
    y_0 = 1
    iterations = 10
    x = 2
    euler_method(t_0, y_0, iterations, x)

    # Runge-Kutta Function
    t_0 = 0
    y_0 = 1
    iterations = 10
    x = 2
    runge_kutta(t_0, y_0, iterations, x)

    #Gauss Elimination & Backwards Substitution Functions
    matrix = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]])
    gauss_elimination(matrix)

    #LU factorization Function
    matrix_2 = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype=np.double)
    LU_factorization(matrix_2)

    #Diagonally dominate Function
    n = 5
    dominate_matrix = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
    diagonally_dominate(dominate_matrix, n)

    #Positive definite Function
    definite_matrix = np.matrix([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
    positive_definite(definite_matrix)
