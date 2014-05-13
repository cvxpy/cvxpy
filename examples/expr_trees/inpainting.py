from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

l = misc.lena()
l = l.astype(np.float64, copy=False)
l = l/np.max(l) #rescale pixels into [0,1]

plt.imshow(l, cmap=plt.cm.gray)
#plt.show()

from PIL import Image, ImageDraw

num_lines = 5
width = 5
imshape = l.shape

def drawRandLine(draw,width):
    x = [np.random.randint(0,im.size[0]) for i in range(2)]
    y = [np.random.randint(0,im.size[1]) for i in range(2)]
    xy = zip(x,y)
    #fill gives the color
    draw.line(xy,fill=255,width=width)

im = Image.new("L",imshape)
draw = ImageDraw.Draw(im)
for i in range(num_lines):
    drawRandLine(draw,width)
del draw
# im.show()

err = np.asarray(im,dtype=np.bool)
r = l.copy()
r[err] = 1.0
plt.imshow(r, cmap=plt.cm.gray)

import itertools
idx2pair = np.nonzero(err)
idx2pair = zip(idx2pair[0].tolist(), idx2pair[1].tolist())
pair2idx = dict(itertools.izip(idx2pair, xrange(len(idx2pair))))
idx2pair = np.array(idx2pair) #convert back to numpy array

import scipy.sparse as sp
from cvxopt import spmatrix

def involvedpairs(pairs):
    ''' Get all the pixel pairs whose gradient involves an unknown pixel.
        Input should be a set or dictionary of pixel pair tuples
    '''
    for pair in pairs: #loop through unknown pixels
        yield pair

        left = (pair[0],pair[1]-1)
        if left[1] >= 0 and left not in pairs: #if 'left' in picture, and not already unknown
            yield left

        top = (pair[0]-1,pair[1])
        topright = (pair[0]-1,pair[1]+1)
        #if not on top boundary, top is fixed, and top not already touched by upper right pixel
        if pair[0] > 0 and top not in pairs and topright not in pairs:
            yield top

def formCOO(pair2idx, img):
    m, n = img.shape
    Is, Js, Vs, bs = [[],[]], [[],[]], [[],[]], [[],[]]
    row = 0

    for pixel1 in involvedpairs(pair2idx):
        bottom = (pixel1[0]+1,pixel1[1])
        right= (pixel1[0],pixel1[1]+1)

        for i, pixel2 in enumerate([bottom, right]):

            if pixel2[0] >= m or pixel2[1] >= n:
                bs[i].append(0)
                continue

            b = 0
            for j, pix in enumerate([pixel2, pixel1]):
                if pix in pair2idx: #unknown pixel
                    Is[i].append(row)
                    Js[i].append(pair2idx[pix])
                    Vs[i].append(pow(-1,j))
                else: #known pixel
                    b += pow(-1,j)*img[pix]
            bs[i].append(b)

        row += 1

    '''
        Form Gx and Gy such that the x-component of the gradient is Gx*x + bx,
        where x is an array representing the unknown pixel values.
    '''
    m = len(bs[0])
    n = len(pair2idx)

    Gx = spmatrix(Vs[1], Is[1], Js[1],(m,n))
    Gy = spmatrix(Vs[0], Is[0], Js[0],(m,n))

    bx = np.array(bs[1])
    by = np.array(bs[0])

    return Gx, Gy, bx, by


Gx, Gy, bx, by = formCOO(pair2idx, r)
import cvxpy as cp
m, n = Gx.size
x = cp.Variable(n)

#z = cp.vstack((x.__rmul__(Gx) + bx).T, (x.__rmul__(Gy) + by).T)
#z = cp.hstack(x.__rmul__(Gx) + bx, x.__rmul__(Gy) + by)
z = cp.Variable(m, 2)
constraints = [z[:, 0] == x.__rmul__(Gx) + bx,
               z[:, 1] == x.__rmul__(Gy) + by]

objective = cp.Minimize(sum([cp.norm(z[i,:]) for i in range(m)]))
p = cp.Problem(objective, constraints)
import cProfile
cProfile.run("""
result = p.solve(solver=cp.ECOS, verbose=True)
""")

