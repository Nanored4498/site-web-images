import numpy as np
from PIL import Image

tanh2 = lambda z: (1 if z.real > 7e2 else (-1 if z.real < -7e2 else np.tanh(z)))
inv_sinh2 = lambda z: (0 if abs(z.real) > 7e2 else 1 / np.sinh(z))
f = lambda z: z - 1 / tanh2(z) + inv_sinh2(z)

ks = [0]*601
mat, m2 = [], [] 
for y in np.arange(np.pi-0.38, np.pi+0.38, 0.0005):
	l, l2 = [], [] 
	for x in np.arange(-0.3, 0.3, 0.0005): 
		z, k = x+y*1j, 0 
		while abs(z.real) > 0.00045 and abs(round(z.imag/(2*np.pi)) - z.imag/(2*np.pi)) > 0.0005 and k < 600:
			z, k = f(z), k+1
		ks[k] += 1
		l.append(k) 
		l2.append(round(z.imag / (2*np.pi))) 
	mat.append(l) 
	m2.append(l2) 

w, h = len(mat[0]), len(mat) 
im = np.zeros((h, w, 3), dtype=np.uint8) 
mi2, ma, ma2 = min(min(l) for l in m2), max(max(l) for l in mat), max(max(l) for l in m2) 
for x in range(w): 
	for y in range(h):
		i = mat[y][x]
		r = (0.4*i if i < 2 else 0.2*i+0.15) if i<7 else min(3.4, 1.4-30**0.47+(max(0, i-7)+30)**0.47)
		im[y, x] = (int(255*r/3.4), int(255*(m2[y][x]/ma2)**0.25) if m2[y][x] > 0 else 0, int(255*(m2[y][x]/mi2)**0.25) if m2[y][x] < 0 else 0)
Image.fromarray(im).convert("RGB").save("test.png")                                                                                                  

print(ks)
import pylab as pl
pl.figure(1)
pl.plot([(0.4*i if i < 2 else 0.2*i+0.15) if i<7 else min(3.4, 1.4-30**0.47+(max(0, i-7)+30)**0.47) for i in range(601)], ks)
pl.figure(2)
pl.plot([i for i in range(601)], ks)
pl.show()