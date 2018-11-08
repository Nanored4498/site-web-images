#!/usr/bin/env python3
from base import pointSetCircle, pointSet, setToLists
from convexe import convex_hull
import pylab as pl
import matplotlib.animation as anim

s = pointSetCircle(0, 0, 1, 80) + pointSet(-1.4, -0.65, -1.4, -0.65, 16)
chs = convex_hull(s, True)

fig = pl.figure()
xs, ys = setToLists(s)
pl.scatter(xs, ys, linewidths=0.0001)
chs = [setToLists(ch) for ch in chs]
cl, = pl.plot([], [], color='red', linewidth=2)

def animate(i):
	j = i % len(chs)
	cl.set_data(chs[j][0], chs[j][1])
	return cl,

ani = anim.FuncAnimation(fig, animate, repeat=True, interval=80)
pl.show()