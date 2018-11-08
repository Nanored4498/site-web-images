from base import positiveOrientation

class Elem:
	def __init__(self, v, pre, nex):
		self.v = v
		self.pred = pre
		self.next = nex
	
	def __str__(self):
		s = "None" if self.pred == None else str(self.pred.v)
		s += " - " + str(self.v) + " - "
		s += "None" if self.next == None else str(self.next.v)
		return s
	
	def copy(self):
		return Elem(self.v, self.pred, self.next)

def circleList(l):
	u = Elem(l[0], None, None)
	v = u
	for i in range(1, len(l)):
		w = Elem(l[i], v, None)
		v.next = w
		v = w
	v.next = u
	u.pred = v
	return u

def toList(cl):
	a = cl
	r = [a.v]
	b = cl.next
	while b != a:
		r.append(b.v)
		b = b.next
	r.append(a.v)
	return r

def convex_hull(s, steps=False):
	s.sort()
	u = circleList([s[2], s[1], s[0]] if positiveOrientation(s[2], s[1], s[0]) else [s[2], s[0], s[1]])
	if steps:
		sts = [toList(u)]
	for i in range(3, len(s)):
		v = s[i]
		w = u.copy()
		while not positiveOrientation(v, u.v, u.next.v):
			u = u.next
		e = Elem(v, None, u)
		u.pred = e
		while positiveOrientation(v, w.v, w.pred.v):
			w = w.pred
		e.pred = w
		w.next = e
		w.pred.next = w
		u = e
		if steps:
			sts.append(toList(u))
	if steps:
		return sts
	return toList(u)