from cvxpy import *
from itertools import izip, imap
import cvxopt
import pylab

X, Y = 'x', 'y'

# create simple image
n = 10
img = cvxopt.matrix(0.0,(n,n))
img[3:9,3:9] = 0.5

# add noise
img = img + 0.1*cvxopt.uniform(n,n)

# show the image
plt = pylab.imshow(img)
plt.set_cmap('gray')
pylab.show()

# define the gradient functions
def grad(img, direction):
    m, n = img.size
    for i in range(m):
        for j in range(n):
            if direction is Y and j > 0 and j < m-1:
                yield img[i,j+1] - img[i,j-1]
            if direction is X and i > 0 and i < n-1:
                yield img[i+1,j] - img[i-1,j]
            yield 0


# take the gradients
img_gradx, img_grady = grad(img,'x'), grad(img,'y')

# filter them (remove ones with small magnitude)
def denoise(grad, thresh):
    for g in grad:
         if g*g >= thresh*thresh: yield g
         else: yield 0.0

denoise_gradx, denoise_grady = denoise(img_gradx, 0.1), denoise(img_grady, 0.1)

# function to get boundary of image
def boundary(img):
    m, n = img.size
    for i in range(m):
        for j in range(n):
            if i == 0 or j == 0:
                yield img[i,j]

# now, reconstruct the image by solving a least-squares problem
new_img = Variable(n,n)
gradx_obj = imap(square, (fx - gx for fx, gx in izip(grad(new_img,'x'),denoise_gradx)))
grady_obj = imap(square, (fy - gy for fy, gy in izip(grad(new_img,'y'),denoise_grady)))
import cProfile
cProfile.run('''
exp1 = vstack(*gradx_obj)
exp2 = vstack(*grady_obj)
obj = cvxopt.matrix(1,exp1.size).trans() * exp1 + \
              cvxopt.matrix(1,exp2.size).trans() * exp2

p = Problem(
    Minimize( obj ),
    list(px == 0 for px in boundary(new_img))
)
p.solve()
''')
#
# # show the reconstructed image
# plt = pylab.imshow(new_img)
# plt.set_cmap('gray')
# pylab.show()