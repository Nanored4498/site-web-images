
from random import random
import math
import numpy as np

class Vec2D:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	
	def __sub__(self, v):
		return Vec2D(self.x - v.x, self.y - v.y)
	
	def __add__(self, v):
		return Vec2D(self.x + v.x, self.y + v.y)
	
	def __mul__(self, v):
		return Vec2D(self.x * v, self.y * v)
	def __rmul__(self, v):
		return Vec2D(self.x * v, self.y * v)

	def __truediv__(self, v):
		return Vec2D(self.x / v, self.y / v)
	
	def __lt__(self, v):
		return self.x < v.x or (self.x == v.x and self.y < v.y)
	
	def __str__(self):
		return "(" + str(round(self.x, 3)) + ", " + str(round(self.y, 3)) + ")"
	
	__repr__ = __str__

	def __eq__(self, other):
		return type(other) == Vec2D and abs(self.x - other.x) < 1e-5 and abs(self.y - other.y) < 1e-5
	
	def __hash__(self):
		return hash((self.x, self.y))

	def dot(self, v):
		return v.x * self.x + v.y * self.y
	
	def norm1(self):
		return abs(self.x) + abs(self.y)
	
	def norm22(self):
		return self.x ** 2 + self.y ** 2

	def norm2(self):
		return self.norm22() ** 0.5

def turnPositive(v):
	return Vec2D(-v.y, v.x)

def positiveOrientation(a, b, c):
	return turnPositive(b - a).dot(c - b) >= 0

def det2(a, b):
	return a.x*b.y - a.y*b.x

def interLine(a0, a1, b0, b1):
	a = a1-a0
	b = b1-b0
	d = det2(a, b)
	if d == 0:
		return None
	da = det2(a0, a1)
	db = det2(b0, b1)
	return (db*a - da*b) / d

def insideConv(p, P):
	i, j = 1, len(P)-2
	while j-i > 1:
		mid = (j+i) // 2
		if positiveOrientation(P[0], P[mid], p):
			i = mid
		else:
			j = mid
	return positiveOrientation(P[0], P[i], p) and positiveOrientation(P[i], P[j], p) and positiveOrientation(P[j], P[0], p)

def interConv(P, Q):
	i, j = 1, 1
	inters = []
	stopP, stopQ = False, False
	while True:
		if i == len(P):
			stopP = True
			i = 1
			if stopQ:
				break
		if j == len(Q):
			stopQ = True
			j = 1
			if stopP:
				break
		a = P[i] - P[i-1]
		b = Q[j] - Q[j-1]
		a_left_of_b = positiveOrientation(Q[j-1], Q[j], P[i])
		a_point_left_b = turnPositive(b).dot(a) > 0
		b_left_of_a = positiveOrientation(P[i-1], P[i], Q[j])
		a_point_b = a_left_of_b != a_point_left_b
		b_point_a = b_left_of_a == a_point_left_b
		if a_point_b:
			if b_point_a:
				if a_left_of_b:
					j += 1
				else:
					i += 1
			else:
				i += 1
		else:
			if b_point_a:
				j += 1
			else:
				inter = interLine(P[i-1], P[i], Q[j-1], Q[j])
				min_x = max(min(P[i-1].x, P[i].x), min(Q[j-1].x, Q[j].x))
				max_x = min(max(P[i-1].x, P[i].x), max(Q[j-1].x, Q[j].x))
				min_y = max(min(P[i-1].y, P[i].y), min(Q[j-1].y, Q[j].y))
				max_y = min(max(P[i-1].y, P[i].y), max(Q[j-1].y, Q[j].y))
				if inter != None and min_x <= inter.x <= max_x and min_y <= inter.y <= max_y:
					inters.append((i, j, inter, a_left_of_b))
				if a_left_of_b:
					j += 1
				else:
					i += 1
	if inters == []:
		if insideConv(P[0], Q):
			return P.copy()
		else:
			return Q.copy()
	inters.append(inters[0])
	res = []
	for i in range(len(inters)-1):
		a, b, inter, is_p = inters[i]
		res.append(inter)
		if is_p:
			na = inters[i+1][0]
			while a != na:
				res.append(P[a])
				a += 1
				if a == len(P):
					a = 1
		else:
			nb = inters[i+1][1]
			while b != nb:
				res.append(Q[b])
				b += 1
				if b == len(Q):
					b = 1
	res.append(res[0])
	return res


def centroid(P):
	res = Vec2D(0, 0)
	mass = 0
	for i in range(1, len(P)-2):
		t = Triangle(P[0], P[i], P[i+1], compute_mass=True)
		res += t.centroid * t.area
		mass += t.area
	return res / mass

def pointSet(x0, x1, y0, y1, n):
	s = []
	for _ in range(n):
		x = x0 + (x1-x0) * random()
		y = y0 + (y1-y0) * random()
		s.append(Vec2D(x, y))
	return s

def pointSetSpace(x0, x1, y0, y1, n, e):
	s, i = [], 0
	while i < n:
		x = x0 + (x1-x0) * random()
		y = y0 + (y1-y0) * random()
		v = Vec2D(x, y)
		b = True
		for j in range(i):
			if (s[j] - v).norm1() < e:
				b = False
				break
		if b:
			s.append(v)
			i += 1
	return s

def pointSetCircle(x, y, r, n):
	s = []
	for _ in range(n):
		p = math.sqrt(random()) * r
		t = math.pi * 2 * random()
		s.append(Vec2D(x + p * math.cos(t), y + p * math.sin(t)))
	return s

def setToLists(s):
	x, y = [], []
	for p in s:
		x.append(p.x)
		y.append(p.y)
	return np.array(x), np.array(y)

def setToRectangle(s):
	x0, x1 = min(p.x for p in s), max(p.x for p in s)
	y0, y1 = min(p.y for p in s), max(p.y for p in s)
	return x0, x1, y0, y1

def densify(s, m):
	res = []
	for i in range(len(s)-1):
		for j in range(m):
			res.append((s[i]*(m-j) + s[i+1]*j) / m)
	res.append(s[-1])
	return res

class Triangle:
	def __init__(self, a, b, c, compute_cent=False, compute_mass=False):
		self.a = a
		self.b = b
		self.c = c
		if compute_cent:
			self.compute_center()
		if compute_mass:
			self.compute_centroid()
	
	def compute_center(self):
		ac = self.c - self.a					# Vector AC
		dac = ac.dot((self.c-self.b)) / 2
		pab = turnPositive(self.b-self.a)
		d = ac.dot(pab)
		self.center = (self.a + self.b) / 2 + dac / d * pab
		v = self.center - self.a
		self.r2 = v.dot(v)
	
	def compute_centroid(self):
		self.centroid = (self.a+self.b+self.c)/3
		self.area = (self.c-self.a).dot(turnPositive(self.b - self.a)) / 2

	def insideCircle(self, p):
		v = p - self.center
		return v.dot(v) < self.r2
	
	def radius(self):
		return self.r2 ** 0.5

	def edges(self):
		return [(self.a, self.b), (self.a, self.c), (self.b, self.c)]
	
	def hasEdge(self, e):
		a, b = e
		es = self.edges()
		return e in es or (b, a) in es

	def vertices(self):
		return [self.a, self.b, self.c]
	
	def hasVertice(self, v):
		return v == self.a or v == self.b or v == self.c
	
	def toPylab(self):
		return setToLists(self.vertices() + [self.a])

class UnionFind:
	def __init__(self, s):
		self.p, self.size = {}, {}
		for x in s:
			self.p[x] = x
			self.size[x] = 1
	
	def root(self, x):
		while self.p[x] != x:
			self.p[x] = self.p[self.p[x]]
			x = self.p[x]
		return x
	
	def find(self, x, y):
		return self.root(x) == self.root(y)

	def union(self, x, y):
		x = self.root(x)
		y = self.root(y)
		if self.size[x] > self.size[y]:
			self.p[y] = x
			self.size[x] += self.size[y]
		else:
			self.p[x] = y
			self.size[y] += self.size[x]

def bezier(ps, num_ims, num_points_between_ims):
	res = []
	t = 0
	dt = 1 / num_ims / num_points_between_ims
	pp = ps[0]
	for _ in range(num_ims):
		im = []
		points = [pp]
		for j in range(num_points_between_ims):
			t += dt
			ps2 = ps
			while len(ps2) > 1:
				nps = []
				for k in range(len(ps2)-1):
					nps.append(t*ps2[k+1] + (1-t)*ps2[k])
				ps2 = nps
				if j == num_points_between_ims-1:
					im.append(ps2)
			points.append(ps2[0])
		pp = points[-1]
		res.append((im, points))
	return res
				