#!/usr/bin/env python3
from base import pointSetSpace, setToLists
import pylab as pl
import matplotlib.animation as anim
from delaunay import delaunay

n = 35
s = pointSetSpace(0, 1, 0, 1, n, 0.8 / n**0.5)

fig = pl.figure()
xs, ys = setToLists(s)
pl.scatter(xs, ys, linewidths=1.8)
tss = delaunay(s, True)
mts = 0
for ts in tss:
	mts = max(mts, len(ts))
cts = []
for _ in range(mts):
	c, = pl.plot([], [], linewidth=1.3)
	cts.append(c)
ltss = len(tss)
m = ltss + 8

def animate(i):
	j = i % m
	if j >= ltss:
		return
	ts = tss[j]
	for k in range(len(ts)):
		x, y = ts[k]
		cts[k].set_data(x, y)
	for k in range(len(ts), mts):
		cts[k].set_data([], [])

ani = anim.FuncAnimation(fig, animate, repeat=True, interval=220)
pl.show()