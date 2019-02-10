import pylab as pl
import base
import delaunay
from random import random
import plot_opt
import math
import matplotlib.animation as anim

monk0 = [(93, 455), (46, 447), (19, 441), (12, 433), (10, 416), (19, 396), (34, 375), (52, 353), (80, 321), (72, 279), (68, 223), (64, 210), (79, 161), (93, 145), (172, 102), (191, 92), (232, 77), (266, 73), (332, 72), (360, 75), (411, 43), (440, 39), (464, 46), (484, 64), (491, 78), (500, 86), (492, 125), (499, 148), (500, 168), (496, 178), (484, 193), (464, 198),(405, 176), (389, 171), (377, 182), (389, 205), (408, 263), (421, 306), (437, 358), (456, 403), (460, 432), (466, 449), (456, 457), (431, 456), (422, 451), (426, 447), (431, 444),(430, 438), (427, 428), (422, 396), (377, 430), (350, 414), (346, 406), (360, 388), (364, 392), (364, 403), (369, 408), (376, 410), (392, 390), (386, 372), (383, 357), (372, 341),(308, 270), (297, 255), (293, 236), (250, 248), (196, 251), (211, 274), (227, 303), (234, 331), (245, 358), (259, 396), (269, 418), (289, 420), (322, 419), (340, 430), (340, 441),(334, 442), (329, 449), (289, 467), (259, 473), (238, 459), (203, 418), (179, 361), (178, 353), (142, 329), (132, 354), (103, 383), (66, 409), (77, 415), (110, 426), (117, 439), (113, 443), (101, 445), (100, 452), (93, 455)]
monk0 = [base.Vec2D(u, -v) for u, v in monk0]
monk = base.densify(monk0, 6)
cirs = []

# Initialisation du dessein
x0, x1, y0, y1 = base.setToRectangle(monk)
w, h = x1-x0, y1-y0
fig = pl.figure()
ax = fig.gca()
pl.xlim(x0-0.1*w, x1+0.1*w)
pl.ylim(y0-0.1*h, y1+0.1*h)
x, y = base.setToLists(monk0)
pl.plot(x, y)
cirs_im = []


sp = [base.Vec2D(2*x0-x1, 2*y0-y1), base.Vec2D(2*x1-x0, 2*y0-y1), base.Vec2D(2*x0-x1, 2*y1-y0), base.Vec2D(2*x1-x0, 2*y1-y0)]
ts = delaunay.delaunay(monk[:-1] + sp)
def valid(c, p):
	for a, r in cirs:
		if (p - a).norm22() < r:
			return False
	count = 0
	for i in range(len(monk0)-1):
		a, b = monk0[i], monk0[i+1]
		if (p.y > a.y) != (p.y > b.y):
			x = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x)
			if x < p.x:
				count += 1
	return count % 2 == 1
nstep = 22
otherpoints = []
ims = []
csp = {}
inp = {}

# Boucle
for step in range(nstep):
	# Squellette du singe
	im = cirs_im.copy()
	sk = set()
	cs = delaunay.voronoi(otherpoints[:-1], ts, sp)
	for c in cs:
		ch = cs[c]
		for i in range(len(ch)-1):
			if valid(c, ch[i]) and valid(c, ch[i+1]):
				sk.add(ch[i])
				sk.add(ch[i+1])
				tu = (ch[i], ch[i+1])
				if tu in inp:
					im.append(inp[tu])
				else:
					x, y = base.setToLists(ch[i:i+2])
					plo = pl.plot(x, y, "r")[0]
					im.append(plo)
					inp[tu] = plo
		if c in csp and cs[c] == csp[c]:
			im.append(csp[c])
		else:
			x, y = base.setToLists(cs[c])
			plo = pl.plot(x, y, "r", linewidth=0.1)[0]
			csp[c] = plo
			im.append(plo)
	ims.append(im)

	# Distance au singe
	ds = {}
	for c in cs:
		for p in cs[c]:
			if p in sk:
				if p in ds:
					ds[p] = min(ds[p], (c-p).norm22())
				else:
					ds[p] = (c-p).norm22()

	# Meilleur centre
	p0, d0 = None, 0
	for p in sk:
		if ds[p] > d0:
			p0, d0 = p, ds[p]
	cirs.append((p0, d0))
	d0 = (d0 ** 0.5) * 0.99
	circ = pl.Circle((p0.x, p0.y), d0, color=(random()*0.8, random()*0.8, random()*0.8))
	if step < nstep-1:
		cirs_im.append(ax.add_artist(circ))
	print("ok", p0, d0)

	# Ajout du nouveau contour
	otherpoints = []
	NC, ang = 36, 0
	da = 2*math.pi / NC
	for i in range(NC+1):
		x = math.cos(ang)
		y = math.sin(ang)
		ang += da
		otherpoints.append(p0 + d0 * base.Vec2D(x, y))

ani = anim.ArtistAnimation(fig, [[], []] + ims, interval=450, repeat=True)
ani.save('res.gif', writer='imagemagick')
# pl.show()