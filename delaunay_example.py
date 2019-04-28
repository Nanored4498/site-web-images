#!/usr/bin/env python3
from base import pointSetSpace, setToLists
import pylab as pl
import matplotlib.animation as anim
from delaunay import delaunay
import plot_opt
from random import random

n = 32
s = pointSetSpace(0, 1, 0, 1, n, 0.8 / n**0.5)
s.sort(key=lambda p: -abs(p.x-0.5) - abs(p.y-0.5) + random())

fig = pl.figure()
tss = delaunay(s, True)

ims = []
pts = {}
x, y = setToLists(s)
ps = [pl.scatter(x, y, color='green', s=10)]
toc = lambda t: (0, 0.5-0.5*min(1, max(0, t.center.x)), 0.5+0.5*min(1, max(0, t.center.y)))
for i in range(n):
	p = pl.scatter([s[i].x], [s[i].y], color='black', s=64)
	ps.append(p)
	im = ps.copy()
	for t in tss[i]:
		if t in pts:
			im.append(pts[t])
		else:
			x, y = t.toPylab()
			im.append(pl.plot(x, y, color='red')[0])
			if t in tss[i+1]:
				pts[t] = pl.plot(x, y, color=toc(t))[0]
	ims.append(im)
	if 12 < i < 17:
		ims.append(im)
im = ps.copy()
for t in tss[-1]:
	im.append(pts[t])
ims += [im] * 5

ani = anim.ArtistAnimation(fig, ims, interval=220, repeat=True)
ani.save('res.gif', writer='imagemagick')
# pl.show()