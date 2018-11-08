
from random import random
import math

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
		return "(" + str(round(self.x, 2)) + ", " + str(round(self.y, 2)) + ")"
	
	def dot(self, v):
		return v.x * self.x + v.y * self.y
	
	def norm1(self):
		return abs(self.x) + abs(self.y)
	
	def norm2(self):
		return (self.x ** 2 + self.y ** 2) ** 0.5

def turnPositive(v):
	return Vec2D(-v.y, v.x)

def positiveOrientation(a, b, c):
	return turnPositive(b - a).dot(c - b) >= 0

def pointInPolygon(p, P):
	for i in range(len(P)-1):
		if not positiveOrientation(P[i], P[i+1], p):
			return False
	return True

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
		p = math.sqrt(random())
		t = math.pi * 2 * random()
		s.append(Vec2D(p * math.cos(t), p * math.sin(t)))
	return s

def setToLists(s):
	x, y = [], []
	for p in s:
		x.append(p.x)
		y.append(p.y)
	return x, y

class Triangle:
	def __init__(self, a, b, c):
		self.a = a
		self.b = b
		self.c = c
		ac = c - a
		dac = ac.dot((c-b)) / 2
		pab = turnPositive(b-a)
		d = ac.dot(pab)
		self.center = (a + b) / 2 + dac / d * pab
		v = self.center - a
		self.r2 = v.dot(v)

	def insideCircle(self, p):
		v = p - self.center
		return v.dot(v) < self.r2
	
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
