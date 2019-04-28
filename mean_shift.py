import plot_opt
from base import pointSet, setToLists, pointSetCircle, pointSetSpace, Vec2D
import pylab as pl
import matplotlib.animation as anim
import math
import clustering

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
clusters = []
k_mean_ims = []

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
	if ni % (N // 7) == 0:
		if len(centers) < NC:
			clustering.add_uniform_center(ps, ds, centers)
		clusters = clustering.k_mean_step(centers, ps)
		k_mean_ims = []
		for i in range(len(centers)):
			x, y = setToLists(clusters[i])
			x += 1
			center = centers[i]
			col = (center.x, abs(center.x - center.y), center.y)
			center += Vec2D(1, 0)
			k_mean_ims.append(pl.scatter(center.x, center.y, color=col, marker='x', s=[100]))
			k_mean_ims.append(pl.scatter(x, y, color=col))
	im += k_mean_ims
	ims.append(im)

ani = anim.ArtistAnimation(fig, ims, interval = 100, repeat=True)
#ani.save('res.gif', writer='imagemagick')
pl.show()