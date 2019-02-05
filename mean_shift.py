import plot_opt
from base import pointSet, setToLists, pointSetCircle, pointSetSpace, Vec2D
import pylab as pl
import matplotlib.animation as anim
import math
from random import random

NC = 3
noise = pointSet(0.02, 0.98, 0, 1, 22)
centers = pointSetSpace(0.25, 0.75, 0.2, 0.8, NC, 1.3 / NC)
circs = [pointSetCircle(c.x, c.y, 0.22, 25) for c in centers]
ps = noise
for c in circs:
	ps += c
lines = [[p] for p in ps]
fig = pl.figure(figsize=(10, 5))
ims = []

N = 36
md = 0.02 ** 2
sig2 = 0.02
inter = 0.15
lenp = len(ps)
ds = [10] * lenp
centers = []

vertl, = pl.plot([1, 1], [0, 1], color='black')
t1, t2 = pl.text(0.05, 0.98, "Mean-Shift"), pl.text(1.05, 0.98, "K-Mean")
for ni in range(N):
	im = [vertl, t1, t2]
	x, y = setToLists(ps)
	for i in range(lenp):
		l = lines[i]
		x, y = setToLists(l)
		im.append(pl.plot(x, y, color='red')[0])
		lp = l[-1]
		c = (lp.x, abs(lp.x - lp.y), lp.y)
		im.append(pl.scatter([ps[i].x], [ps[i].y], color=c))
	if ni != N-1:
		np = []
		for l in lines:
			lp = l[-1]
			s = 0
			p = Vec2D(0, 0)
			for l2 in lines:
				if l == l2:
					continue
				k = math.exp(-(lp - l2[-1]).norm22() / sig2)
				s += k
				p += k * l2[-1]
			np.append((1 - inter)* lp + inter * (p / s))
		for i in range(len(lines)):
			lines[i].append(np[i])
	#####################################################
	if ni % (N // 7) == 0 and len(centers) < NC:
		s = sum(ds)
		dsn = [d / s for d in ds]
		r = random()
		i = 0
		while r > dsn[i]:
			r -= dsn[i]
			i += 1
		centers.append(ps[i])
		for j in range(lenp):
			ds[j] = min(ds[j], (ps[i] - ps[j]).norm22())
	ncs = len(centers)
	clus = [[] for _ in range(ncs)]
	sc = [Vec2D(0, 0) for _ in range(ncs)]
	for p in ps:
		i = -1
		d = 10
		for j in range(ncs):
			nd = (p - centers[j]).norm22()
			if nd < d:
				i, d = j, nd
		clus[i].append(p + Vec2D(1, 0))
		sc[i] += p
	for i in range(ncs):
		x, y = setToLists(clus[i])
		nc = sc[i] / len(clus[i])
		if ni % (N // 7) == 0 and ncs == NC:
			centers[i] = nc
		col = (nc.x, abs(nc.x - nc.y), nc.y)
		im.append(pl.scatter(nc.x+1, nc.y, color=col, marker='x', s=[100]))
		im.append(pl.scatter(x, y, color=col))
	ims.append(im)

ani = anim.ArtistAnimation(fig, ims, interval = 100, repeat=True)
ani.save('res.gif', writer='imagemagick')
#pl.show()