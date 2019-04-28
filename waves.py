import plot_opt
import pylab as pl
import numpy as np
import matplotlib.animation as anim

# x = np.arange(0, 10, 0.02)
# y = np.cos(x)
# for i in range(100):
# 	y[i] = y[-i-1] = 0
# yp = [0]*len(x)

x, y = np.meshgrid(np.arange(-85.0, 85.0, 1.0), np.arange(-85.0, 85.0, 1.0))
z = np.sqrt((x*x + y*y))
zp = 0.0*x
for i in range(170):
	for j in range(170):
		if z[i, j] > 13: z[i, j] = 0
		else:
			zp[i, j] = np.cos(2.0*np.pi/13.0 * z[i][j])
			z[i, j] = np.sin(2*np.pi/13 * z[i][j])

def Lap1D(y):
	y2 = [y[1] - y[0]]
	for i in range(1, len(y)-1):
		y2.append(y[i+1]+y[i-1]-2*y[i])
	y2.append(y[-2] - y[-1])
	return np.array(y2)

def Lap2D(y):
	(n, m) = y.shape
	yt = y.T
	y1 = []
	for i in range(n):
		y1.append(Lap1D(y[i]))
	y2 = []
	for j in range(m):
		y2.append(Lap1D(yt[j]))
	return np.array(y1) + np.array(y2).T

dt = 0.3

# pl.contourf(x, y, z)
# pl.show()

fig = pl.figure()
ims = [list(pl.contourf(x, y, z).collections)]
# print(list(pl.contourf(x, y, z).collections))
# exit(0)
for step in range(120):
	for _ in range(10):
		# y = y + dt * Lap1D(y)
		# yp = yp + dt * Lap1D(y)
		# y = y + dt*yp
		zp = zp + dt * Lap2D(z)
		z = z + dt*zp
	# ims.append([pl.plot(x, y, color='blue')[0]])
	ims.append(list(pl.contourf(x, y, z).collections))
	print(step)

ani = anim.ArtistAnimation(fig, ims, interval=100, repeat=True)
pl.show()