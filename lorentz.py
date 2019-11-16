import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import plot_opt

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0
N = 10
Nframes = 50
wait = 6

def f(state, t):
  x, y, z = state
  return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

state0 = (np.random.rand(N, 3) - [[0.5, 0.5, 0.2]]) * [[45, 45, 60]]
t = np.arange(0.0, 8.0, 0.005)

fig = pl.figure()
ax = fig.gca(projection='3d')
ax.set_xlim3d([-30, 30])
ax.set_ylim3d([-30, 30])
ax.set_zlim3d([-10, 60])

states, curves, ps = [], [], []
for s0 in state0:
	s = odeint(f, s0, t).T
	states.append(s)
	c = ax.plot(s[0,:], s[1,:], s[2,:])[0]
	p = ax.scatter([s[0,0]], [s[1,0]], [s[2,0]], color=c.get_color())
	ps.append(p)
	curves.append(c)

def update_curves(num, states, curves, ps):
	for s, c, p in zip(states, curves, ps):
		n = s.shape[1]
		l = min(n, int(1+n * (num / (Nframes-1))**2.5))
		c.set_data(s[:2, :l])
		c.set_3d_properties(s[2, :l])
		p._offsets3d = ([s[0,l-1]], [s[1,l-1]], [s[2,l-1]])
	return curves, p

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_curves, Nframes+wait, fargs=(states, curves, ps),
                                   interval=90, blit=False)
line_ani.save('test.gif', writer='imagemagick')

pl.show()