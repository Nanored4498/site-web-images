import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from math import cos, sin, pi, log
import matplotlib.animation as anim
import plot_opt

im_width, im_height = 576, 448
N = 280
N = N // 2
r = abs(complex(0.123, 0.745))
zabs_max = 3
nit_max = 7038
xmin, xmax = -1.25, 1.25
xwidth = xmax - xmin
ymin, ymax = xmin * im_width / im_height, xmax * im_width / im_height
yheight = ymax - ymin
fig, ax = pl.subplots(figsize=(im_width / 100, im_height / 100))
po = log(0.775, 2) / log(32 / nit_max, 2)
macma = ((nit_max-1) / nit_max) ** po
di = (1 - (20 / nit_max)**po) / (0.875 - (20 / nit_max)**po)
print(po, di)

julias = [0]*(N*2)
for i in range(N+1):
	print(i * 100 / N)
	th = i * pi / N
	c = r * complex(cos(th), sin(th))
	julia = np.zeros((im_height, im_width))
	j2 = np.zeros((im_height, im_width))
	mc, mac = 1, 0
	for ix in range(im_height):
		for iy in range(im_width):
			nit = 0
			z = complex(iy / im_width * yheight + ymin, ix / im_height * xwidth + xmin)
			while abs(z) <= zabs_max and nit < nit_max:
				z = z**2 + c
				nit += 1
			ratio = nit / nit_max
			j = ratio ** po
			julia[ix,iy] = j
			j2[im_height-1-ix,iy] = j
			mc = min(mc, j)
			mac = max(mac, j)
	im = pl.imshow(julia, animated=True, cmap=cm.jet)
	im.cmap.set_over('black')
	im.set_clim(mc, (((di-1)*mac**2 + macma**2) / di) ** 0.5)
	julias[i] = [im]
	if i > 0 and i < N:
		im2 = pl.imshow(j2, animated=True, cmap=cm.jet)
		im2.cmap.set_over('black')
		im2.set_clim(mc, (((di-1)*mac**1.8625 + macma**1.8625) / di) ** (1 / 1.8625))
		julias[2*N-i] = [im2]
	

ani = anim.ArtistAnimation(fig, julias, interval=128, repeat=True, blit=True)
ani.save('ani15.gif', writer='imagemagick')

# pl.show()
