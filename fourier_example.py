#!/usr/bin/env python3

import pylab as pl
import plot_opt
import math
import cmath
import matplotlib.animation as anim

def sym(ps):
	l = len(ps)
	for i in range(l-1):
		a, b = ps[l-2-i]
		ps.append((2.8 - a, b))
ps = [(0, 0), (-1, 1), (-2, 2), (-2.2, 3), (-2, 4), (-1, 5), (0.3, 5.5),
	  (0, 6.6), (-0.6, 7.6), (-1.3, 8.5), (-1.8, 7.7), (-1.65, 6.8), (-1.25, 5.7), (-1, 5),
	  (-0.15, 5.32), (-0.55, 6.1), (-0.9, 6.05), (-0.65, 5.15), (0.3, 5.5), (1.4, 5.7)]
sym(ps)
ps2 = [(2.8, 2), (2.6, 2.7), (2.3, 3.2), (2, 3), (1.8, 2.5), (1.7, 2), (1.7, -0.45),
	   (2.8, 0), (2.4, -0.15), (2.4, 0.5), (2.2, 0.8),
	   (2.1, 0.5), (2.05, -0.3), (1.5, -0.56), (1.4, -0.56)]
sym(ps2)
ps += ps2 + [(0, 0)]

minX, maxX = 1000, -1000
minY, maxY = 1000, -1000
for a, b in ps:
	minX = min(minX, a)
	maxX = max(maxX, a)
	minY = min(minY, b)
	maxY = max(maxY, b)
ad = 0.42
w, h = maxX - minX + 2*ad, maxY - minY + 2*ad
co = 42 / w / h
fig = pl.figure(figsize=(w*co, h*co))
pl.xlim(minX - ad, maxX + ad)
pl.ylim(minY - ad, maxY + ad)

cps = [complex(a, b) for a, b in ps]
def add(cps):
	ncps = []
	for i in range(len(cps)-1):
		ncps += [cps[i], (cps[i] + cps[i+1]) / 2]
	return ncps + [cps[-1]]
for _ in range(4):
	cps = add(cps)

T = len(cps)-1
N = 60
c = []
ns = list(range(-N+1, N))
for n in ns:
	su = 0
	v = 1
	mv = cmath.exp(- 2 * math.pi * 1j * n / T)
	for i in range(T):
		su += cps[i] * v
		v *= mv
	c.append(su / T)

c2 = []
for i in range(len(c)):
	if ns[i] == 0:
		pos0 = c[i]
	else:
		c2.append((c[i], ns[i]))
c2.sort(key=lambda x: -abs(x[0]))
mr = abs(c2[0][0])

def circle(center, size):
	nps = max(8, int(64 * size / mr))
	x0, y0 = center.real, center.imag
	cirX, cirY = [], []
	for i in range(nps):
		cirX.append(x0 + size * math.cos(2 * math.pi * i / nps))
		cirY.append(y0 + size * math.sin(2 * math.pi * i / nps))
	cirX.append(cirX[0])
	cirY.append(cirY[0])
	return pl.plot(cirX, cirY, color='red', linewidth=0.5)[0]

x, y = [], []
np = 500
temps = 6400
fps = 17
inter = np / (temps * fps / 1000 * 0.85)
t = inter
ims = []
for i in range(np):
	if t >= inter or i == np-1:
		t -= inter
		show = True
		im = []
	else:
		show = False
	p = pos0
	en = [cmath.exp(2 * math.pi * 1j * ns[0] * i / np)]
	mul = cmath.exp(2 * math.pi * 1j * i / np)
	for j in range(len(ns)-1):
		en.append(en[j] * mul)
	for coef, n in c2:
		p2 = p + en[n - ns[0]] * coef
		if show:
			im.append(pl.plot([p.real, p2.real], [p.imag, p2.imag], color='purple')[0])
			im.append(circle(p, abs(coef)))
		p = p2
	x.append(p.real)
	y.append(p.imag)
	t += 1
	if show:
		im.append(pl.plot(x, y, color='blue')[0])
		ims.append(im)
	print(i)
x.append(x[0])
y.append(y[0])
ims += [[pl.plot(x, y, color='blue')[0]]] * int(0.15 * temps * fps / 1000)

ani = anim.ArtistAnimation(fig, ims, interval = temps // len(ims), repeat=True)
ani.save('res.gif', writer='imagemagick')