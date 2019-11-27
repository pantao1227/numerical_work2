#coding=utf-8

import numpy as np
from numpy import float64

# 下面函数用于生成希尔伯特系数矩阵
def hilb(n):
    h=np.zeros((n,n), dtype=float64)
    for i in range(1, n+1):
        for j in range(1, n+1):
            h[i-1,j-1] = 1.0 / (i + j - 1)
    return h


# 下面函数用于使用高斯-赛德尔迭代法解方程
def s_Gauss_Seidel(A,b,x,eps,N):
    n = b.size
    for k in range(1,N+1):
        x0 = x.copy()   # 记录上一个X的值
        for i in range(0,n):
            t = 0.0 # 重置临时变量
            for j in range(0,n):
                if j != i :
                    t = t + A[i][j] * x[j]
            x[i] = (b[i] - t) / A[i][i]
        if np.linalg.norm(x-x0,ord=2) < eps :
            break
    return (k,x)


# 下面函数用于使用最速下降法解方程
def s_Steepest_Descent(A,b,x,eps,N):
    for k in range(1,N+1):
        r = b - np.matmul(A,x)
        t = np.matmul(r.T,r)/np.matmul(np.matmul(r.T,A),r)
        x = x + t * r
        if np.linalg.norm(r, ord=2) < eps :
            break
    return (k,x)


# 下面函数用于使用共轭梯度法（原始CG法）解方程
def s_Conjugate_Gradient(A,b,x,eps,N):
    r = b - np.matmul(A,x)
    p = r.copy()
    for k in range(1,N+1):
        alpha = np.dot(r, r) / np.dot(np.matmul(A,p), p)
        x = x + alpha * p
        r = r - alpha * np.matmul(A,p)
        beta = - np.dot(r, np.matmul(A, p)) / np.dot(p, np.matmul(A, p))
        p = r + beta * p
        if np.linalg.norm(r,ord=2) < eps:
            break
    return [k, x]


A = hilb(16)
b = np.array(
    [2877/851, 3491/1431, 816/409, 2035/1187, 
    2155/1423, 538/395, 1587/1279, 573/502, 
    947/895, 1669/1691, 1589/1717, 414/475, 
    337/409, 905/1158, 1272/1711, 173/244], 
    dtype = float64
    ).T
x = np.zeros(b.size, dtype = float64).T

print(s_Gauss_Seidel(A, b, x.copy(), 0.001, 1000000))

# from collections import defaultdict

# result = defaultdict(dict)

# for eps in [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]:
#     result['Gauss_Seidel'][eps] = s_Gauss_Seidel(A, b, x.copy(), eps, 10000)
#     result['Steepest_Descent'][eps] = s_Steepest_Descent(A, b, x.copy(), eps, 10000)
#     result['Conjugate_Gradient'][eps] = s_Conjugate_Gradient(A, b, x.copy(), eps, 10000)

# for name in ['Gauss_Seidel','Steepest_Descent','Conjugate_Gradient']:
#     print('{}'.format(name))
#     for eps in [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]:
#         print(
#             '{:<8}:{:>8}\n{}'.format(eps, result[name][eps][0], result[name][eps][1])
#         )
# print("\n")
