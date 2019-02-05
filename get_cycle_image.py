import matplotlib.pyplot as plt

fig = plt.figure()
po, = plt.plot([], [])
xs, ys = list(po.get_xdata()), list(po.get_ydata())
def onclick(event):
	if event.button == 1:
		if event.dblclick:
			v = []
			for i in range(len(xs)):
				v.append((xs[i], ys[i]))
			v.append((xs[0], ys[0]))
			print(v)
		else:
			xs.append(event.xdata)
			ys.append(event.ydata)
			po.set_data(xs + [xs[0]], ys + [ys[0]])
			po.figure.canvas.draw()

connection_id = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()