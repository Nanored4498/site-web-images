#!/usr/bin/env python3

from base import pointSetSpace, setToLists, Vec2D
from delaunay import toGraph
from graphs import dijkstra, plotGraph
import pylab as pl
import plot_opt
import matplotlib.animation as anim
import numpy as np

n = 180
V = [Vec2D(0.5, 0.5)] + pointSetSpace(0, 1, 0, 1, n-1, 0.8 / n**0.5)
E = toGraph(V)
ds, pred = dijkstra(V, E, V[0])

fig = pl.figure()
pl.xlim(0, 1)
pl.ylim(0, 1)
plotGraph(V, E)

es = []
for v in V[1:]:
	es.append((ds[v], v))
es.sort()
es = [setToLists([e[1], pred[e[1]]]) for e in es]
ls = [pl.plot([], [], color='red', linewidth=1.6)[0] for _ in range(n-1)]

nc = 7
dd = 1 / (nc - 1)
x, y = np.mgrid[slice(0, 1 + dd/2, dd), slice(0, 1 + dd/2, dd)]
z = [[[0, 0] for _ in range(nc)] for _ in range(nc)]
for v in V:
	a = round(nc * (v.x - 1 / nc / 2))
	b = round(nc * (v.y - 1 / nc / 2))
	c = z[a][b]
	c[0] += ds[v]
	c[1] += 1
for l in z:
	for i in range(nc):
		su, na = l[i]
		l[i] = su / na
pl.contourf(x, y, z)

def animate(i):
	j = i % (n + n // 5)
	if j >= n:
		return
	if j == 0:
		for l in ls:
			l.set_data([], [])
	else:
		j -= 1
		x, y = es[j]
		ls[j].set_data(x, y)

ani = anim.FuncAnimation(fig, animate, repeat=True, interval=5)
pl.show()