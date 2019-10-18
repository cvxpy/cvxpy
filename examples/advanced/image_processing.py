"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy import *
import cvxopt
import pylab
import math

# create simple image
n = 32
img = cvxopt.matrix(0.0,(n,n))
img[1:2,1:2] = 0.5

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
            if direction == 'y' and j > 0 and j < m-1:
                yield img[i,j+1] - img[i,j-1]
            elif direction == 'x' and i > 0 and i < n-1:
                yield img[i+1,j] - img[i-1,j]
            else:
                yield 0.0

# take the gradients
img_gradx, img_grady = grad(img,'x'), grad(img,'y')

# filter them (remove ones with small magnitude)

def denoise(gradx, grady, thresh):
    for dx, dy in zip(gradx, grady):
         if math.sqrt(dx*dx + dy*dy) >= thresh: yield (dx,dy)
         else: yield (0.0,0.0)

denoise_gradx, denoise_grady = zip(*denoise(img_gradx, img_grady, 0.2))

# function to get boundary of image
def boundary(img):
    m, n = img.size
    for i in range(m):
        for j in range(n):
            if i == 0 or j == 0 or i == n-1 or j == n-1:
                yield img[i,j]

# now, reconstruct the image by solving a constrained least-squares problem
new_img = Variable(n,n)
gradx_obj = map(square, (fx - gx for fx, gx in zip(grad(new_img,'x'),denoise_gradx)))
grady_obj = map(square, (fy - gy for fy, gy in zip(grad(new_img,'y'),denoise_grady)))

p = Problem(
    Minimize(sum(gradx_obj) + sum(grady_obj)),
    list(px == 0 for px in boundary(new_img)))
p.solve()

# show the reconstructed image
plt = pylab.imshow(new_img.value)
plt.set_cmap('gray')
pylab.show()

print(new_img.value)
