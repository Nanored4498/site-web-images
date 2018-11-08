import heapq as hq
import pylab as pl
from base import setToLists,UnionFind
from delaunay import delaunay

def dijkstra(V, E, o):
	pred = {o:None}
	ds = dict.fromkeys(V, float('inf'))
	ds[o] = 0
	n, m = len(V), 0
	h = [(0, o)]
	hq.heapify(h)
	while m < n:
		d, v = hq.heappop(h)
		if d > ds[v]:
			continue
		m += 1
		for w, dw in E[v]:
			nd = d + dw
			if nd < ds[w]:
				ds[w] = nd
				pred[w] = v
				hq.heappush(h, (nd, w))
	return ds, pred

def plotGraph(V, E, ec='green', ealph=0.5, ew=0.8):
	x, y = setToLists(V)
	pl.scatter(x, y)
	seen = dict.fromkeys(V, False)
	def aux(s):
		seen[s] = True
		for r, _ in E[s]:
			x, y = setToLists([s, r])
			pl.plot(x, y, color=ec, alpha=ealph, linewidth=ew)
			if not seen[r]:
				aux(r)
	aux(V[0])

def pointsToGraph(s):
	ts = delaunay(s)
	E = set([])
	for t in ts:
		for e in t.edges():
			a, b = e
			if e not in E and (b, a) not in E:
				E.add(e)
	E = list(E)
	for i in range(len(E)):
		a, b = E[i]
		v = b-a
		E[i] = (v.dot(v), a, b)
	return (s, E)

def spanning(V, E, clust=1):
	n = len(V)
	E.sort()
	vs = UnionFind(V)
	res = []
	for _, a, b in E:
		if not vs.find(a, b):
			res.append([a, b])
			vs.union(a, b)
			if len(res) == n-clust:
				break
	return res if clust == 1 else vs