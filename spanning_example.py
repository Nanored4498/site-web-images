#!/usr/bin/env python3

from base import pointSet, setToLists
from graphs import spanning
from delaunay import toGraph
import pylab as pl
import plot_opt
import matplotlib.animation as anim

n = 220
s = pointSet(0, 1, 0, 1, n)
E = toGraph(s)
sp = spanning(s, E)

fig = pl.figure()
x, y = setToLists(s)
pl.scatter(x, y, linewidths=0.6)
es = []
for i in range(n-1):
	sp[i] = setToLists(sp[i])
	e, = pl.plot([], [], color='red', linewidth=1.8)
	es.append(e)

def animate(i):
	j = i % (n + n // 5)
	if j >= n:
		return
	if j == 0:
		for e in es:
			e.set_data([], [])
	for k in range(j):
		x, y = sp[k]
		es[k].set_data(x, y)

ani = anim.FuncAnimation(fig, animate, repeat=True, interval=25)
pl.show()