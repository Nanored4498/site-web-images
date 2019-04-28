from random import random
from base import Vec2D

def add_uniform_center(ps, ds, centers):
	s = sum(ds)
	dsn = [d / s for d in ds]
	r = random()
	i = 0
	while r > dsn[i]:
		r -= dsn[i]
		i += 1
	centers.append(ps[i])
	for j in range(len(ps)):
		ds[j] = min(ds[j], (ps[i] - ps[j]).norm22())

def k_mean_step(centers, ps):
	ncs = len(centers)
	clus = [[] for _ in range(ncs)]
	sc = [Vec2D(0, 0) for _ in range(ncs)]
	for p in ps:
		i = -1
		d = float('inf')
		for j in range(ncs):
			nd = (p - centers[j]).norm22()
			if nd < d:
				i, d = j, nd
		clus[i].append(p)
		sc[i] += p
	for i in range(ncs):
		centers[i] = sc[i] / len(clus[i])
	return clus