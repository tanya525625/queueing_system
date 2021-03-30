import numpy as np
import matplotlib.pyplot as plt

n = 3
lmd = np.ones([n+1])
mu = np.ones([n+1])
p_0 = np.zeros([n+1])
p_0[0] = 1
# p_0 = p_0.reshape(-1, 1)
t_0 = 0
t = 10
# Матрица для уравнения Колмагорова
def matrix(lmd, mu, n):
    f = np.zeros([n+1,n+1])
    for i in range(n+1):
        for j in range(n+1):
            if j == i-1:
                f[i,j]= lmd[j-1]
            if j == i:
                f[i, j] = -(lmd[j-1]+i*mu[j-1])
            if j == i+1:
                f[i, j] = (i+1)*mu[i]

    return f
def F(f,p):
    F = np.dot(f, p)
    return F
# Один шаг РК
def one_step(f, p, h):
  k1 = F(f, p)
  k2 = F(f, p+h/2*k1)
  k3 = F(f, p+h/2*k2)
  k4 = F(f, p+h*k3)
  return (p+h/6*(k1+2*k2+2*k3+k4))

def rk4(f, t_0, p_0, t, h=0.1):
  tV = np.arange(t_0, t+h, h)
  tV[-1] = t
  p = np.empty((len(tV), len(p_0)))
  p[0] = p_0
  for i, index in enumerate(tV[: -2]):
    p[i+1] = one_step(f, p[i], h)
  h = tV[-1] - tV[-2]
  p[-1] = one_step(f, p[-2], h)
  return p

f = matrix(lmd, mu, n)
p = rk4(f, t_0, p_0, t, 0.1)
print(p)
