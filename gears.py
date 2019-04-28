import plot_opt
import pylab as pl
import numpy as np
import matplotlib.animation as anim

N = 300
ths = [2*np.pi * i / N for i in range(N+1)]
M = 7
a, p = np.random.rand(M), np.random.rand(M) * 2*np.pi
d = sum(a)
x, y = [], []
x2, y2 = [], []
phs = [0]
rs2 = []
adds = 0
lr = -1
sr = 0
for th in ths:
	r = sum([a[i] * np.cos(i * th / 2 + p[i]) ** 2 for i in range(M)])
	sr += r**2
	r2 = d-r
	if lr != -1:
		add = (lr + r) / (r2 + d - lr) * 2 * np.pi / N
		adds += add
		phs.append(phs[-1] - add)
	lr = r
	rs2.append(r2)

rat = adds / 2 / np.pi
phs = np.pi + np.array(phs) / rat
rs2 = np.array(rs2) * rat
ths = np.array(ths)
d = lr + rs2[-1]
rs = d - rs2

fig = pl.figure()
pl.scatter([0, d], [0, 0])
ims = []
NI = 150
for j in range(NI):
	i = int(j * N / NI)
	x = np.cos(ths - ths[i])*rs
	y = np.sin(ths - ths[i])*rs
	x2 = d + np.cos(phs - phs[i] + np.pi)*rs2
	y2 = np.sin(phs - phs[i] + np.pi)*rs2
	ims.append([pl.plot(x, y, "b")[0], pl.plot(x2, y2, "r")[0]])

ani = anim.ArtistAnimation(fig, ims, interval=60, repeat=True)
# ani.save('res.gif', writer='imagemagick')
pl.show()