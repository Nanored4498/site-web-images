from base import pointSet, setToLists, bezier
import pylab as pl
import plot_opt
import matplotlib.animation as anim

nim = 45
npoint = 500
temps = 3200
s = pointSet(0, 1, 0, 1, 5)
s.sort()
print(s)
bez = bezier(s, nim, int(npoint / nim))
colors = ['green', 'blue', 'purple', 'black']

fig = pl.figure()
piece_curve = []
x, y = setToLists(s)
init_points = pl.scatter(x, y, color='black')
init_curv = pl.plot(x, y, color='black')[0]
ims = [[init_curv, init_points]]

for cs, points in bez:
	im = [init_curv, init_points]
	x, y = setToLists(points)
	pc = pl.plot(x, y, color='red')[0]
	piece_curve.append(pc)
	for a in piece_curve:
		im.append(a)
	for i in range(len(cs)):
		x, y = setToLists(cs[i])
		im.append(pl.plot(x, y, color=colors[i])[0])
		im.append(pl.scatter(x, y, color=colors[i]))
	ims.append(im)

ani = anim.ArtistAnimation(fig, ims, interval = temps / nim, repeat=True)
ani.save('res.gif', writer='imagemagick')
# pl.show()