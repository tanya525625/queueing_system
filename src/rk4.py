#Implement the method RK 4 order for an arbitrary system of ODE
import numpy as np
import matplotlib.pyplot as plt


def f(t, x, g=9.8, l=1):
  theta, ang_vel  = x[0], x[1]
  return np.array([ang_vel, -g*np.sin(theta)/l])


def one_step(F,index, x,  dt):
  k1 = F(index, x)
  k2 = F(index+dt/2, x +dt/2*k1)
  k3 = F(index+dt/2, x+dt/2*k2)
  k4 = F(index+dt, x+dt*k3)
  return (x+dt/6*(k1+2*k2+2*k3+k4))


def rk4(F, t0, x0, t, dt=0.1):
  index = 0
  tV = np.arange (t0, t+dt, dt)
  tV[-1] = t
  x = np.empty((len(tV), len(x0)))
  x[0] = x0
  for i, index in enumerate(tV[: -2]):
    x[i+1] = one_step (F, index, x[i], dt)
  dt = tV[-1] - tV[-2]
  x[-1] = one_step (F, index, x[-2], dt)
  return x


y1 = np.arange (0, 100.1, 0.1)
y2 = np.arange (0, 100.37, 0.37)
y3 = np.arange (0, 101, 1)
y4 = np.arange (0, 110, 10)

xx1 = [np.pi/4, 0.1]

x1 = rk4(f, 0, xx1, 100, dt=0.1)
x2 = rk4(f, 0, xx1, 100, dt=0.37)
x3 = rk4(f, 0, xx1, 100, dt=1)
x4 = rk4(f, 0, xx1, 100, dt=10)

plt.plot(y1, x1[:, 0] , 'b-', alpha=1)
plt.plot(y2, x2[:, 0], 'r-', alpha=1)
plt.plot(y3, x3[:, 0], 'g-', alpha=1)
plt.plot(y4, x4[:, 0], 'y-', alpha=1)
plt.savefig('1.png')