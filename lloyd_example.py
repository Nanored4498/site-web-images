import plot_opt
from base import Vec2D
import base
import pylab as pl
import delaunay
from random import random
import matplotlib.animation as anim

fig = pl.figure(figsize=(6, 6))

n = 70
N = 40
centers = base.pointSet(0.4, 0.6, 0.4, 0.6, n) + [Vec2D(0.01, 0.01), Vec2D(0.99, 0.01), Vec2D(0.01, 0.99), Vec2D(0.99, 0.99)]
n += 4
colors = [(random(), random(), random()) for _ in range(n)]
square = [Vec2D(0, 0), Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0)]
ims = []
for _ in range(N):
	im = []
	x0, y0 = base.setToLists(centers)
	cells = delaunay.voronoi(centers)
	# centers = [base.centroid(base.interConv(cells[c], square)) for c in centers]
	# cells = delaunay.voronoi(centers)
	for i in range(n):
		c = centers[i]
		cells[c] = base.interConv(cells[c], square)
		x, y = base.setToLists(cells[c])
		im.append(pl.fill(x, y, color=colors[i], linewidth=0, alpha=0.7)[0])
		centers[i] = base.centroid(cells[c])
	im.append(pl.scatter(x0, y0, color="black"))
	ims.append(im)

ani = anim.ArtistAnimation(fig, ims, interval=150, repeat=True)
# ani.save("res.gif", writer='imagemagick')
pl.show()
