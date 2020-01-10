from scipy import ndimage, misc, fftpack
from skimage import color
import numpy as np
import pylab as pl
import imageio

### PARAMETERS ###
K = [32, 64, 16, 128]
##################

# Load
content = pl.imread("./base_images/moulin.jpg")
style = pl.imread("./base_images/starry_night.jpg")
w0, h0, c0 = content.shape
w2, h2, c = style.shape

# Convert to LAB
content = color.rgb2lab(content)
style = color.rgb2lab(style)

# Zoom
zoom_factor = np.sqrt((w0*h0) / (w2*h2))
style = np.array([ndimage.zoom(style[:,:,i], zoom_factor) for i in range(c)])
style = style.transpose((1, 2, 0))

# Make shape multiple of 8
def shape_padding(im, pad):
	w, h, _ = im.shape
	ax, ay = -w % pad, -h % pad
	if ax > 0:
		ll = np.flip(im[-ax:, :], 0)
		im = np.concatenate((im, ll), 0)
	if ay > 0:
		lc = np.flip(im[:, -ay:], 1)
		im = np.concatenate((im, lc), 1)
	return im

k = 1
for ki in K:
	k *= ki // np.math.gcd(k, ki)
content = shape_padding(content, k)
style = shape_padding(style, k)

# DCT
def dct2(im, K):
	res = np.zeros(im.shape)
	for i in np.r_[:im.shape[0]:K]:
		for j in np.r_[:im.shape[1]:K]:
			res[i:(i+K), j:(j+K), :] = fftpack.dct(fftpack.dct(im[i:(i+K), j:(j+K), :], axis=0, norm='ortho'), axis=1, norm='ortho')
	return res

# Optimal Transport
def ot(src, dest, nsteps, ndir):
	n, d = src.shape
	if dest.shape[0] != n:
		np.random.shuffle(dest)
		if dest.shape[0] > n: dest = dest[:n,:]
		else: dest = np.concatenate((dest, dest[:n-dest.shape[0], :]), 0)
	for _ in range(nsteps):
		add = np.zeros(src.shape)
		for _ in range(ndir):
			v = np.random.standard_normal(d)
			v /= np.linalg.norm(v)
			psrc = src.dot(v)
			pdest = dest.dot(v)
			isrc = np.argsort(psrc)
			idest = np.argsort(pdest)
			ind = np.empty(n, isrc.dtype)
			ind[isrc] = np.arange(n)
			add += v * (pdest[idest] - psrc[isrc])[ind].reshape(n, 1)
		src += add / ndir

# iDCT
def idct2(im, K):
	res = np.zeros(im.shape)
	for i in np.r_[:im.shape[0]:K]:
		for j in np.r_[:im.shape[1]:K]:
			res[i:(i+K), j:(j+K), :] = fftpack.idct(fftpack.idct(im[i:(i+K), j:(j+K), :] , axis=0, norm='ortho' ), axis=1, norm='ortho')
	return res

# transfert
def trans(content0, style0, K):
	# DCT
	content = dct2(content0, K)
	style = dct2(style0, K)
	# OT
	w, h, c = content.shape
	w2, h2, c = style.shape
	for i in range(K):
		for j in range(K):
			src = content[i:w:K, j:h:K, :c].reshape((w*h)//(K*K), c)
			dest = style[i:w2:K, j:h2:K, :c].reshape((w2*h2)//(K*K), c)
			ot(src, dest, 20, 2)
			content[i:w:K, j:h:K, :c] = src.reshape(w//K, h//K, c)
	# iDCT
	content = idct2(content, K)
	return content

content0 = content
for k in K:
	content = trans(0.5*(content+content0), style, k)

# Convert to RGB
content = np.minimum(style.max((0, 1)), np.maximum(style.min((0, 1)), content[:w0, :h0, :c]))
content = color.lab2rgb(content)
imageio.imwrite('test.jpg', content)