#!/usr/bin/env python3

from base import pointSet, setToLists
from convexe import convex_hull
import pylab as pl
import plot_opt
import matplotlib.animation as anim
from math import log, exp

T = 80
s = pointSet(0, 1, 0, 1, 6)
x0, y0 = setToLists(convex_hull(s))
l = len(x0)-1
ims = []

fig = pl.figure()

def f(t):
	return (1 + (t / T)**0.45) / 2

di0 = ((x0[0]-x0[1])**2 + (y0[0]-y0[1])**2)**0.5
for t in range(T):
	im = [pl.plot(x0, y0, color=(t/T, 0, 1-t/T))[0]]
	x, y = x0.copy(), y0.copy()
	di = di0 * (1 - f(t))
	d = di0
	while d >= 2*di:
		a = di / d
		for j in range(l):
			x[j] = a*x[j] + (1-a)*x[j+1]
			y[j] = a*y[j] + (1-a)*y[j+1]
		d = ((x[0]-x[1])**2 + (y[0]-y[1])**2)**0.5
		x[-1], y[-1] = x[0], y[0]
		im.append(pl.plot(x, y, color=(t/T, 0, 1-t/T), linewidth=0.75)[0])
	ims.append(im)
	print(t / T, len(im))

ani = anim.ArtistAnimation(fig, ims, interval=33, repeat=True)
ani.save('res.gif', writer='imagemagick')
# pl.show()