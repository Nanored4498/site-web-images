import numpy as np
import matplotlib.image as mpimg
from delaunay import delaunay
from base import *
import pylab as pl
from convexe import convex_hull
import plot_opt

im0 = 1 - mpimg.imread("images/lena.png").mean(2)
im1 = 1 - mpimg.imread("images/peppers.png").mean(2)
assert im0.shape == im1.shape
w, h = im0.shape
coeff = im1.sum() / im0.sum()

n = 1700
nSteps = 25
alpha = 0.0175
beta = 0.05
e = 0.3 * (w+h) / n**0.5
s = pointSetSpace(0, w-1, 0, h-1, n, e) + [Vec2D(-2*w, -2*h), Vec2D(3*w, -2*h), Vec2D(-2*w, 3*h), Vec2D(3*w, 3*h)]

D = delaunay(s)
neighboors = {p:set() for p in s}
for t in D:
	for a, b in t.edges():
		neighboors[a].add(b)
		neighboors[b].add(a)
g = {p:0 for p in s}
s = s[:-4]
neighboors = {p:sorted(neighboors[p], key=lambda q: (q-p).angle()) for p in s}
square = [Vec2D(0, 0), Vec2D(w, 0), Vec2D(w, h), Vec2D(0, h), Vec2D(0, 0)]

def getCol(x, y, P, im, toSee):
	val = 0
	w, h = im.shape
	Q = [(x, y)]
	toSee[x,y] = False
	n = 0
	while len(Q) > 0:
		x, y = Q.pop()
		val += im[x,y]
		n += 1
		for a, b in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
			if a >= 0 and a < w and b >=0 and b < h and toSee[a,b] and insideConv(Vec2D(x, y), P):
				toSee[a,b] = False
				Q.append((a, b))
	return val/n*area(interConv(P, square))

def voronoi(s, g, neighboors):
	Vs = []
	for p in s:
		lines = []
		for q in neighboors[p]:
			pq = q - p
			pq /= pq.norm2()
			o = (p+q)/2 + (g[p]-g[q])*pq
			lines.append((o, pq))
		lines.append(lines[0])
		ps = []
		for i in range(len(lines)-1):
			o, v = lines[i]
			d = turnPositive(v)
			o2, v2 = lines[i+1]
			proj_o = o.dot(v2)
			proj_d = d.dot(v2)
			proj_o2 = o2.dot(v2)
			t = (proj_o2 - proj_o) / proj_d
			inter = o + t * d
			ps.append(inter)
		ps = convex_hull(ps)
		Vs.append(ps)
	return Vs

def integrate(s, g, neighboors, im):
	Vs = voronoi(s, g, neighboors)
	bs = []
	toSee = np.ones(im.shape, dtype=bool)
	for i in range(len(s)):
		p = s[i]
		val = getCol(round(p.x), round(p.y), Vs[i], im, toSee)
		bs.append(val)
	return Vs, np.array(bs)

def step(s, b, g, neighboors, im, alpha):
	P, c = integrate(s, g, neighboors, im)
	add = alpha * (b*coeff - c)
	for i in range(len(s)):
		g[s[i]] += add[i]
	return P

def show_vor(P, b, n, coeff):
	for i in range(n):
		y, x = setToLists(P[i])
		a = max(b[i], coeff*area(interConv(P[i], square)))
		pl.fill(x, y, color=1 - b[i]/a*np.ones(3))

pl.xlim(0, w)
pl.ylim(h, 0)
b = integrate(s, g, neighboors, im0)[1]
for i in range(nSteps):
	print("Step", i)
	P = step(s, b, g, neighboors, im1, alpha/(1+beta*i))
	show_vor(P, b, n, (nSteps-1-i + i/coeff)/(nSteps-1))
	pl.savefig(f"sdot2_{i}.jpg")