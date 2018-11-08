from PIL import Image
from math import exp

def convo(im, x, y, M):
	n = len(M)
	n2 = (n-1)/2
	x0, y0 = x-n2, y-n2
	res = [0, 0, 0]
	for i in range(n):
		for j in range(n):
			c = im.getpixel((x0+i, y0+j))
			for k in range(3):
				res[k] += M[i][j] * c[k]
	return res

def cont(im):
	w, h = im.width, im.height
	print(w, h)
	MH = [[-1, 0, 1]]*3
	MV = [[-1]*3, [0]*3, [1]*3]
	res = Image.new('RGB', (w, h))
	r = [[0]*h for _ in range(w)]
	mv = 0
	for x in range(w):
		for y in range(h):
			if x == 0 or x == w-1 or y == 0 or y == h-1:
				continue
			v = 0
			for vi in convo(im, x, y, MH) + convo(im, x, y, MV):
				v += vi*vi
			v = v**0.5
			mv = max(v, mv)
			r[x][y] = v
	for x in range(w):
		for y in range(h):
			c = int(r[x][y] / mv * 256)
			res.putpixel((x, y), (c, c, c))
	return res

def gauss_mat(n, s):
	res = []
	su = 0
	for i in range(2*n+1):
		res.append([])
		for j in range(2*n+1):
			res[i].append(exp(- ((i-n)**2 + (j-n)**2) / s**2))
			su += res[i][j]
	for l in res:
		for i in range(2*n+1):
			l[i] /= su
	return res

def toCorrectPixel(c):
	return (int(c[0]), int(c[1]), int(c[2]))

def apply_conv(im, M):
	n = (len(M) - 1) // 2
	w, h = im.width, im.height
	res = Image.new("RGB", (w, h))
	for x in range(n, w-2*n):
		for y in range(n, h-2*n):
			res.putpixel((x, y), toCorrectPixel(convo(im, x, y, M)))
	return res

def gauss_lis(im, n, s):
	M = gauss_mat(n, s)
	return apply_conv(im, M)
