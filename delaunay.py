from base import Vec2D, Triangle, setToRectangle
from convexe import convex_hull

def topl(ts):
	return [t.toPylab() for t in ts]

def update_triangu(p, ts):
	badT = []
	poly = set()
	for t in ts:
		if t.insideCircle(p):
			for e in t.edges():
				a, b = e
				if e in poly:
					poly.remove(e)
				elif (b, a) in poly:
					poly.remove((b, a))
				else:
					poly.add(e)
			badT.append(t)
	for t in badT:
		ts.remove(t)
	for e in poly:
		ts.add(Triangle(p, e[0], e[1], compute_cent=True))

def remove_tri(ts, bigT):
	badT = []
	for t in ts:
		for v in t.vertices():
			if bigT.hasVertice(v):
				badT.append(t)
				break
	for t in badT:
		ts.remove(t)

def delaunay(s, steps=False):
	x0, x1, y0, y1 = setToRectangle(s)
	W, H = x1 - x0, y1 - y0
	bigT = Triangle(Vec2D(x0 - W*6/10, y0 - H/20), Vec2D(x1 + W*6/10, y0 - H/20), Vec2D(x0 + W/2, y1 + H*11/10), compute_cent=True)
	ts = {bigT}
	if steps:
		sts = []
	for p in s:
		update_triangu(p, ts)
		if steps:
			ts2 = ts.copy()
			remove_tri(ts2, bigT)
			sts.append(ts2)
	remove_tri(ts, bigT)
	if steps:
		return sts + [ts]
	return ts

def toGraph(s):
	ts = delaunay(s)
	E = {}
	for p in s:
		E[p] = set([])
	for t in ts:
		for a, b in t.edges():
			E[a].add(b)
			E[b].add(a)
	for p in s:
		E[p] = list(E[p])
		for i in range(len(E[p])):
			q = E[p][i]
			E[p][i] = (q, (p - q).norm2())
	return E

def voronoi(s, ts=None, sp=[]):
	if ts == None:
		x0, x1, y0, y1 = setToRectangle(s)
		sp = [Vec2D(2*x0-x1, 2*y0-y1), Vec2D(2*x1-x0, 2*y0-y1), Vec2D(2*x0-x1, 2*y1-y0), Vec2D(2*x1-x0, 2*y1-y0)]
		ts = delaunay(s+sp, False)
	else:
		for p in s:
			update_triangu(p, ts)
	cells = {}
	for t in ts:
		c = t.center
		for v in t.vertices():
			if v in cells:
				if c not in cells[v]:
					cells[v].append(c)
			else:
				cells[v] = [c]
	to_remove = []
	for p in cells:
		if p in sp:
			to_remove.append(p)
		else:
			# print(p)
			cells[p] = convex_hull(cells[p])
	for p in to_remove:
		del cells[p]
	return cells