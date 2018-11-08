#!/usr/bin/env python3

from PIL import Image
from graphs import spanning

# im = Image.open("panda.jpg").convert('RGB')
im = Image.open("lisM.png").convert('RGB')
# im = Image.open("lisM2.png").convert('RGB')
h, w = im.height, im.width
V = [i for i in range(w*h)]
E = []

def dc(a, b):
	return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)**0.5

for x in range(w-1):
	for y in range(h-1):
		c = im.getpixel((x, y))
		n = x + y*w
		E.append((dc(c, im.getpixel((x+1, y))), n, n+1))
		E.append((dc(c, im.getpixel((x, y+1))), n, n+w))
print("Graph OK")

uf = spanning(V, E, 65000)
print("Clustering OK")

cs = {}
for x in range(w):
	for y in range(h):
		v = x + y*w
		rv = uf.root(v)
		c = im.getpixel((x, y))
		if rv in cs:
			for k in range(3):
				cs[rv][0][k] += c[k]
			cs[rv][1] += 1
		else:
			cs[rv] = [[c[0], c[1], c[2]], 1]
for r in cs:
	c, s = cs[r]
	cs[r] = (int(c[0] / s), int(c[1] / s), int(c[2] / s)) #if s > 100 else (255, 0, 0)
print("Color OK")

res = Image.new("RGB", (w, h))
for x in range(w):
	for y in range(h):
		v = x + y*w
		rv = uf.root(v)
		res.putpixel((x, y), cs[rv])
res.save("test.png")
print("Save OK")