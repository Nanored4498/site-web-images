from base import Vec2D, Triangle

def topl(ts):
	return [t.toPylab() for t in ts]

def delaunay(s, steps=False):
	x0, x1, y0, y1 = float("inf"), -float("inf"), float("inf"), -float("inf")
	for p in s:
		x0 = min(x0, p.x)
		x1 = max(x1, p.x)
		y0 = min(y0, p.y)
		y1 = max(y1, p.y)
	W, H = x1 - x0, y1 - y0
	bigT = Triangle(Vec2D(x0 - W*6/10, y0 - H/20), Vec2D(x1 + W*6/10, y0 - H/20), Vec2D(x0 + W/2, y1 + H*11/10))
	ts = {bigT}
	if steps:
		sts = [topl(ts)]
	for p in s:
		badT = []
		poly = set([])
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
			ts.add(Triangle(p, e[0], e[1]))
		if steps:
			sts.append(topl(ts))
	badT = []
	for t in ts:
		for v in t.vertices():
			if bigT.hasVertice(v):
				badT.append(t)
				break
	for t in badT:
		ts.remove(t)
	if steps:
		return sts + [topl(ts)]
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