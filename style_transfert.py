from scipy import ndimage, fftpack
from skimage import color
import numpy as np
import pylab as pl
import cv2 as cv

### PARAMETERS ###
K = 8
##################

# Load
content = pl.imread("./base_images/moulin.jpg")
style = pl.imread("./base_images/starry_night.jpg")

# Convert to LAB
# content = np.array([ndimage.zoom(content[:,:,i], 0.4) for i in range(3)]).transpose((1, 2, 0))
content = color.rgb2lab(content)
style = color.rgb2lab(style)

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

def resize_input(content0, style0, K):
	# Padding
	content = shape_padding(content0, K)
	style = shape_padding(style0, K)
	# Style tiling
	while style.shape[0] < content.shape[0]:
		style = np.concatenate((style, style[:content.shape[0]-style.shape[0]]), 0)
	while style.shape[1] < content.shape[1]:
		style = np.concatenate((style, style[:,:content.shape[1]-style.shape[1]]), 1)
	return content, style

# DCT
def dct2(im, K):
	res = im[np.r_[:im.shape[0]:K][:,None,None,None] + np.r_[:K][:,None],
			np.r_[:im.shape[1]:K][:,None,None] + np.r_[:K], :]
	res = fftpack.dct(res, axis=2, norm='ortho', overwrite_x=True)
	res = fftpack.dct(res, axis=3, norm='ortho', overwrite_x=True)
	return res

# iDCT
def idct2(im, K):
	res = fftpack.idct(im, axis=3, norm='ortho')
	res = fftpack.idct(res, axis=2, norm='ortho', overwrite_x=True)
	res = res[np.r_[:im.shape[0]].repeat(K)[:,None], np.r_[:im.shape[1]].repeat(K),
			np.tile(np.r_[:K], im.shape[0])[:,None], np.tile(np.r_[:K], im.shape[1]), :]
	return res

# Optimal Transport
def ot(src, dest, nsteps, ndir):
	d = src.shape[-1]
	K = src.shape[-2]
	ix, iy = np.r_[:K][:,None], np.r_[:K]
	for _ in range(nsteps):
		add = np.zeros(src.shape)
		for _ in range(ndir):
			v = np.random.standard_normal(d)
			v /= np.linalg.norm(v)
			psrc = src.dot(v)
			pdest = dest.dot(v)
			isrc = np.argsort(psrc, axis=0)
			add[isrc, ix, iy] += v * np.expand_dims(np.sort(pdest, axis=0) - psrc[isrc, ix, iy], -1)
		src += add / ndir

# transfert
def trans0(content0, style0, K):
	w0, h0, c0 = content0.shape
	content, style = resize_input(content0, style0, K)
	w, h, c = content.shape
	# DCT
	content = dct2(content, K)
	style = dct2(style, K)
	# OT
	src = content.reshape((w*h)//(K*K), K, K, c)
	dest = style.reshape((w*h)//(K*K), K, K, c)
	ot(src, dest, 20, 2)
	content = src.reshape(w//K, h//K, K, K, c)
	# iDCT
	content = idct2(content, K)
	return content[:w0,:h0,:c0]

def trans(content0, style0, K):
	# Zoom
	zoom_factor = min(content0.shape[0]/style0.shape[0], content0.shape[1]/style0.shape[1])
	style = np.array([ndimage.zoom(style0[:,:,i], zoom_factor) for i in range(style0.shape[2])])
	style = style.transpose((1, 2, 0))
	# Gaussian Pyramid
	GC, GS = [content0], [style]
	while min(min(GC[-1].shape[:2]), min(GS[-1].shape[:2])) >= 4*K:
		GC.append(cv.pyrDown(GC[-1]))
		GS.append(cv.pyrDown(GS[-1]))
	# Laplacian
	LC, LS = [], []
	for l in range(len(GC)-1):
		LC.append(GC[l] - cv.pyrUp(GC[l+1], dstsize=(GC[l].shape[1], GC[l].shape[0])))
		LS.append(GS[l] - cv.pyrUp(GS[l+1], dstsize=(GS[l].shape[1], GS[l].shape[0])))
	LC.append(GC[-1])
	LS.append(GS[-1])
	# transfert and reconstruction
	L = [trans0(lc, ls, K) for lc, ls in zip(LC, LS)]
	res = L[-1]
	for l in range(2, len(L)+1):
		r_up = cv.pyrUp(res, dstsize=(L[-l].shape[1], L[-l].shape[0]))
		res = r_up + L[-l]
	res = np.minimum(style0.max((0, 1)), np.maximum(style0.min((0, 1)), res))
	return res

# Transfert
res = trans(content, style, K)
res = color.lab2rgb(res)
pl.imsave('test.jpg', res)