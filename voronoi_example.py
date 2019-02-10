#!/usr/bin/env python3
from base import pointSetSpace, setToLists, turnPositive, Vec2D, pointInPolygon
import pylab as pl
import matplotlib.animation as anim
from delaunay import delaunay
from convexe import convex_hull
import plot_opt
from PIL import Image
import sys

sys.setrecursionlimit(10**4)

n = 1000
rat = 12 / 8
s = pointSetSpace(0, rat, 0, 1, n, 0.8 / n**0.5)
s += [Vec2D(-10, -10), Vec2D(10, -10), Vec2D(-10, 10), Vec2D(10, -10)]

fig = pl.figure()
pl.xlim(0, rat)
pl.ylim(0, 1)
xs, ys = setToLists(s)
# pl.scatter(xs, ys, linewidths=1, color='blue')
ts = delaunay(s)
cs = []
es = {}
vs = {}
for t in ts:
	#x, y = t.toPylab()
	#pl.plot(x, y, linewidth=0.8, color='green')
	c = t.center
	cs.append(c)
	for e in t.edges():
		a, b = e
	# 	if e in es:
	# 		x, y = setToLists([es[e], c])
	# 		pl.plot(x, y, linewidth=0.4, color='red')
	# 		del es[e]
	# 	else:
	# 		e2 = a, b
	# 		if e2 in es:
	# 			pl.plot(x, y, linewidth=0.4, color='red')	
	# 			pl.plot(x, y)
	# 			del es[e2]
	# 		else:
	# 			es[e] = c
		for p in [a, b]:
			if p in vs:
				if c not in vs[p]:
					vs[p].append(c)
			else:
				vs[p] = [c]
# for e in es:
# 	a, b = e
# 	v = turnPositive(b-a)

im = Image.open("base_images/panda.jpg").convert("RGB")
w, h = im.width, im.height
toSee = [[True]*h for _ in range(w)]

def getCol(x, y, P, first=True):
	col, np = [0, 0, 0], 0
	if first or pointInPolygon(Vec2D(x/w*rat, y/h), P):
		toSee[x][y] = False
		rgb = im.getpixel((x, h-1-y))
		for i in range(3):
			col[i] += rgb[i]
		np = 1
		for a, b in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
			if a >= 0 and a < w and b >=0 and b < h and toSee[a][b]:
				c2, n2 = getCol(a, b, P, False)
				for i in range(3):
					col[i] += c2[i]
				np += n2
	return col, np

sn = 0
for p in vs:
	ps = vs[p]
	if len(ps) > 2:
		P = convex_hull(ps)
		col, np = getCol(max(0, min(w-1, int(p.x*w/rat))), max(0, min(h-1, int(p.y*h))), P)
		sn += np
		for i in range(3):
			col[i] /= np * 255
		x, y = setToLists(P)
		pl.fill(x, y, alpha=0.9, color=col)
print(sn, w*h)
xc, yc = setToLists(cs)
# pl.scatter(xc, yc, linewidths=0.5, color='black')

pl.show()