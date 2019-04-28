import plot_opt
from base import Vec2D
import base
import pylab as pl
import delaunay
from PIL import Image

fig = pl.figure(figsize=(6, 6))

n = 6000
centers = base.pointSet(0, 1, 0, 1, n-4) + [Vec2D(0.01, 0.01), Vec2D(0.99, 0.01), Vec2D(0.01, 0.99), Vec2D(0.99, 0.99)]
square = [Vec2D(0, 0), Vec2D(1, 0), Vec2D(1, 1), Vec2D(0, 1), Vec2D(0, 0)]
im = Image.open("base_images/panda.jpg").convert("RGB")
w, h = im.width, im.height

def get_col(c):
	col = im.getpixel((int(c.x * w), int(h - c.y * h)))
	return tuple(col[i] / 255 for i in range(3))

cells = delaunay.voronoi(centers)

x0, y0 = base.setToLists(centers)
for i in range(n):
	c = centers[i]
	cells[c] = base.interConv(cells[c], square)
	x, y = base.setToLists(cells[c])
	pl.fill(x, y, color=get_col(c), linewidth=0)

pl.show()
