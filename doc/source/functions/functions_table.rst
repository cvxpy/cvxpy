.. |_| unicode:: 0xA0
   :trim:

.. list-table::
   :class: atomic-functions
   :header-rows: 1

   * - Function
     - Meaning
     - Domain
     - DCP Properties
     - Curvature |_|
     - Type

   * - :ref:`dotsort(X,W) <dotsort>`

       constant :math:`W \in \mathbf{R}^{o \times p}`
     - :math:`\text{dot product of}`
       :math:`\operatorname{sort}\operatorname{vec}(X) \text{ and}`
       :math:`\operatorname{sort}\operatorname{vec}(W)`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - sign depends on :math:`X`, :math:`W`
       
       |incr| for :math:`\min(W) \geq 0`
     - |convex| convex
     - scalar

   * - :ref:`geo_mean(x) <geo-mean>`

       :ref:`geo_mean(x, p) <geo-mean>`

       :math:`p \in \mathbf{R}^n_{+}`

       :math:`p \neq 0`
     - :math:`x_1^{1/n} \cdots x_n^{1/n}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - |positive| positive

       |incr| incr.
     - |concave| concave
     - scalar

   * - :ref:`harmonic_mean(x) <harmonic-mean>`
     - :math:`\frac{n}{\frac{1}{x_1} + \cdots + \frac{1}{x_n}}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - |positive| positive

       |incr| incr.
     - |concave| concave
     - scalar

   * - :ref:`inv_prod(x) <inv-prod>`
     - :math:`(x_1\cdots x_n)^{-1}`
     - :math:`x \in \mathbf{R}^n_+`
     - |positive| positive

       |decr| decr.
     - |convex| convex
     - scalar

   * - :ref:`lambda_max(X) <lambda-max>`
     - :math:`\lambda_{\max}(X)`
     - :math:`X \in \mathbf{S}^n`
     - |unknown| unknown sign
     - |convex| convex
     - scalar

   * - :ref:`lambda_min(X) <lambda-min>`
     - :math:`\lambda_{\min}(X)`
     - :math:`X \in \mathbf{S}^n`
     - |unknown| unknown sign
     - |concave| concave
     - scalar

   * - :ref:`lambda_sum_largest(X,k) <lambda-sum-largest>`

       :math:`k = 1,\ldots, n`
     - :math:`\text{sum of $k$ largest}`
       :math:`\text{eigenvalues of $X$}`
     - :math:`X \in\mathbf{S}^{n}`
     - |unknown| unknown sign
     - |convex| convex
     - scalar

   * - :ref:`lambda_sum_smallest(X,k) <lambda-sum-smallest>`

       :math:`k = 1,\ldots, n`
     - :math:`\text{sum of $k$ smallest}`
       :math:`\text{eigenvalues of $X$}`
     - :math:`X \in\mathbf{S}^{n}`
     - |unknown| unknown sign
     - |concave| concave
     - scalar

   * - :ref:`log_det(X) <log-det>`

     - :math:`\log \left(\det (X)\right)`
     - :math:`X \in \mathbf{S}^n_+`
     - |unknown| unknown sign
     - |concave| concave
     - scalar

   * - :ref:`log_sum_exp(X) <log-sum-exp>`

     - :math:`\log \left(\sum_{ij}e^{X_{ij}}\right)`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |unknown| unknown sign

       |incr| incr.
     - |convex| convex
     - scalar

   * - :ref:`matrix_frac(x, P) <matrix-frac>`

     - :math:`x^T P^{-1} x`
     - :math:`x \in \mathbf{R}^n`
     - |positive| positive
     - |convex| convex
     - scalar

   * - :ref:`max(X) <max>`

     - :math:`\max_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same sign as X

       |incr| incr.
     - |convex| convex
     - scalar

   * - :ref:`mean(X) <mean>`

     - :math:`\frac{1}{m n}\sum_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same sign as X

       |incr| incr.
     - |affine| affine
     - scalar

   * - :ref:`min(X) <min>`

     - :math:`\min_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same sign as X

       |incr| incr.
     - |concave| concave
     - scalar

   * - :ref:`mixed_norm(X, p, q) <mixed-norm>`

     - :math:`\left(\sum_k\left(\sum_l\lvert x_{k,l}\rvert^p\right)^{q/p}\right)^{1/q}`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - |positive| positive
     - |convex| convex
     - scalar

   * - :ref:`norm(x) <norm>`

       norm(x, 2)
     - :math:`\sqrt{\sum_{i} \lvert x_{i} \rvert^2 }`
     - :math:`X \in\mathbf{R}^{n}`
     - |positive| positive

       |incr| for :math:`x_{i} \geq 0`
     - |convex| convex
     - scalar

   * - :ref:`norm(x, 1) <norm>`
     - :math:`\sum_{i}\lvert x_{i} \rvert`
     - :math:`x \in\mathbf{R}^{n}`
     - |positive| positive

       |incr| for :math:`x_{i} \geq 0`
     - |convex| convex
     - scalar

   * - :ref:`norm(x, "inf") <norm>`
     - :math:`\max_{i} \{\lvert x_{i} \rvert\}`
     - :math:`x \in\mathbf{R}^{n}`
     - |positive| positive

       |incr| for :math:`x_{i} \geq 0`
     - |convex| convex
     - scalar

   * - :ref:`norm(X, "fro") <norm>`
     - :math:`\sqrt{\sum_{ij}X_{ij}^2 }`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive

       |incr| for :math:`X_{ij} \geq 0`
     - |convex| convex
     - scalar

   * - :ref:`norm(X, 1) <norm>`
     - :math:`\max_{j} \|X_{:,j}\|_1`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive

       |incr| for :math:`X_{ij} \geq 0`
     - |convex| convex
     - scalar

   * - :ref:`norm(X, "inf") <norm>`
     - :math:`\max_{i} \|X_{i,:}\|_1`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
      
       |incr| for :math:`X_{ij} \geq 0`
     - |convex| convex
     - scalar

   * - :ref:`norm(X, "nuc") <norm>`
     - :math:`\mathrm{tr}\left(\left(X^T X\right)^{1/2}\right)`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - scalar

   * - :ref:`norm(X) <norm>`
       norm(X, 2)
     - :math:`\sqrt{\lambda_{\max}\left(X^T X\right)}`
     - :math:`X in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - scalar

   * - :ref:`perspective(f(x),s) <perspective>`

     - :math:`sf(x/s)`
     - :math:`x \in \mathop{\bf dom} f`
       :math:`s \geq 0`
     - same sign as f
     - |convex| / |concave|
       same as :math:`f`
     - scalar

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`p \geq 1`
       or ``p = 'inf'``
     - :math:`\left(\sum_{ij} |X_{ij}|^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |positive| positive

       |incr| for :math:`X_{ij} \geq 0`
     - |convex| convex
     - scalar

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`p < 1`, :math:`p \neq 0`
     - :math:`\left(\sum_{ij} X_{ij}^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}_+`
     - |positive| positive
       
       |incr| incr.
     - |concave| concave
     - scalar

   * - :ref:`ptp(X) <ptp>`

     - :math:`\max_{ij} X_{ij} - \min_{ij} X_{ij}`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - scalar

   * - :ref:`quad_form(x, P) <quad-form>`

       constant :math:`P \in \mathbf{S}^n_+`
     - :math:`x^T P x`
     - :math:`x \in \mathbf{R}^n`
     - |positive| positive

       |incr| for :math:`x_i \geq 0`
     - |convex| convex
     - scalar

   * - :ref:`quad_form(x, P) <quad-form>`

       constant :math:`P \in \mathbf{S}^n_-`
     - :math:`x^T P x`
     - :math:`x \in \mathbf{R}^n`
     - |negative| negative

       |decr| for :math:`x_i \geq 0`
     - |concave| concave
     - scalar

   * - :ref:`quad_form(c, X) <quad-form>`

       constant :math:`c \in \mathbf{R}^n`
     - :math:`c^T X c`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - sign depends |_| on |_| c, |_| X
       
       monotonicity depends |_| on |_| c
     - |affine| affine
     - scalar

   * - :ref:`quad_over_lin(X, y) <quad-over-lin>`

     - :math:`\left(\sum_{ij}X_{ij}^2\right)/y`
     - :math:`x \in \mathbf{R}^n`
       :math:`y > 0`
     - |positive| positive

       |incr| for :math:`X_{ij} \geq 0`
       |decr| for :math:`X_{ij} \leq 0`
       |decr| decr. in :math:`y`
     - |convex| convex
     - scalar

   * - :ref:`std(X) <std>`

     - analog to `numpy.std <https://numpy.org/doc/stable/reference/generated/numpy.std.html#numpy-std>`_
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - scalar

   * - :ref:`sum(X) <sum>`

     - :math:`\sum_{ij}X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same sign as X

       |incr| incr.
     - |affine| affine
     - scalar

   * - :ref:`sum_largest(X, k) <sum-largest>`

       :math:`k = 1,2,\ldots`
     - :math:`\text{sum of } k`
     
       :math:`\text{largest }X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same sign as X

       |incr| incr.
     - |convex| convex
     - scalar

   * - :ref:`sum_smallest(X, k) <sum-smallest>`

       :math:`k = 1,2,\ldots`
     - :math:`\text{sum of } k`
     
       :math:`\text{smallest }X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same sign as X

       |incr| incr.
     - |concave| concave
     - scalar

   * - :ref:`sum_squares(X) <sum-squares>`

     - :math:`\sum_{ij}X_{ij}^2`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
       
       |incr| for :math:`X_{ij} \geq 0`
       |decr| for :math:`X_{ij} \leq 0`
     - |convex| convex
     - scalar

   * - :ref:`trace(X) <trace>`

     - :math:`\mathrm{tr}\left(X \right)`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - same sign as X

       |incr| incr.
     - |affine| affine
     - scalar

   * - :ref:`tr_inv(X) <tr_inv>`

     - :math:`\mathrm{tr}\left(X^{-1} \right)`
     - :math:`X \in\mathbf{S}^n_{++}`
     - |positive| positive
     - |convex| convex
     - scalar

   * - :ref:`tv(x) <tv>`

     - :math:`\sum_{i}|x_{i+1} - x_i|`
     - :math:`x \in \mathbf{R}^n`
     - |positive| positive
     - |convex| convex
     - scalar

   * - :ref:`tv(X) <tv>`
       :math:`Y = \left[\begin{matrix} X_{i+1,j} - X_{ij} \\ X_{i,j+1} -X_{ij} \end{matrix}\right]`
     - :math:`\sum_{ij}\left\| Y \right\|_2`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - scalar

   * - :ref:`tv([X1,...,Xk]) <tv>`
       :math:`Y = \left[\begin{matrix} X_{i+1,j}^{(1)} - X_{ij}^{(1)} \\ X_{i,j+1}^{(1)} -X_{ij}^{(1)} \\ \vdots \\ X_{i+1,j}^{(k)} - X_{ij}^{(k)} \\ X_{i,j+1}^{(k)} -X_{ij}^{(k)}  \end{matrix}\right]`
     - :math:`\sum_{ij}\left\| Y \right\|_2`
     - :math:`X^{(i)} \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - scalar

   * - :ref:`var(X) <var>`

     - analog to `numpy.var <https://numpy.org/doc/stable/reference/generated/numpy.var.html#numpy-var>`_
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - scalar

   * - :ref:`abs(x) <abs>`

     - :math:`\lvert x \rvert`
     - :math:`x \in \mathbf{C}`
     - |positive| positive
       
       |incr| for :math:`x \geq 0`
     - |convex| convex
     - elementwise

   * - :ref:`conj(x) <conj>`

     - complex conjugate
     - :math:`x \in \mathbf{C}`
     - |unknown| unknown sign
     - |affine| affine
     - elementwise

   * - :ref:`entr(x) <entr>`

     - :math:`-x \log (x)`
     - :math:`x > 0`
     - |unknown| unknown sign
     - |concave| concave
     - elementwise

   * - :ref:`exp(x) <exp>`

     - :math:`e^x`
     - :math:`x \in \mathbf{R}`
     - |positive| positive

       |incr| incr.
     - |convex| convex
     - elementwise

   * - :ref:`huber(x, M=1) <huber>`

       :math:`M \geq 0`
     - :math:`\begin{aligned} & \text{if } |x| \leq M\colon \\& x^2 \end{aligned}`
     
       :math:`\begin{aligned} & \text{if } |x| > M\colon \\& 2M|x| - M^2 \end{aligned}`
     - :math:`x \in \mathbf{R}`
     - |positive| positive

       |incr| for :math:`x \geq 0`
       
       |decr| for :math:`x \leq 0`
     - |convex| convex
     - elementwise

   * - :ref:`imag(x) <imag-atom>`

     - imaginary part 
     
       of a complex number
     - :math:`x \in \mathbf{C}`
     - |unknown| unknown sign
     - |affine| affine
     - elementwise

   * - :ref:`inv_pos(x) <inv-pos>`

     - :math:`1/x`
     - :math:`x > 0`
     - |positive| positive
       
       |decr| decr.
     - |convex| convex
     - elementwise

   * - :ref:`kl_div(x, y) <kl-div>`

     - :math:`x \log(x/y) - x + y`
     - :math:`x > 0`

       :math:`y > 0`
     - |positive| positive
     - |convex| convex
     - elementwise

   * - :ref:`log(x) <log>`

     - :math:`\log(x)`
     - :math:`x > 0`
     - |unknown| unknown sign

       |incr| incr.
     - |concave| concave
     - elementwise

   * - :ref:`log_normcdf(x) <log-normcdf>`

     - :ref:`approximate <clarifyelementwise>` log of the standard normal CDF
     - :math:`x \in \mathbf{R}`
     - |negative| negative

       |incr| incr.
     - |concave| concave
     - elementwise

   * - :ref:`log1p(x) <log1p>`

     - :math:`\log(x+1)`
     - :math:`x > -1`
     - same sign as x

       |incr| incr.
     - |concave| concave
     - elementwise

   * - :ref:`loggamma(x) <loggamma>`

     - :ref:`approximate <clarifyelementwise>` `log of the Gamma function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loggamma.html>`_
     - :math:`x > 0`
     - |unknown| unknown sign
     - |convex| convex
     - elementwise

   * - :ref:`logistic(x) <logistic>`

     - :math:`\log(1 + e^{x})`
     - :math:`x \in \mathbf{R}`
     - |positive| positive

       |incr| incr.
     - |convex| convex
     - elementwise

   * - :ref:`maximum(x, y) <maximum>`

     - :math:`\max \left\{x, y\right\}`
     - :math:`x,y \in \mathbf{R}`
     - sign depends on x,y

       |incr| incr.
     - |convex| convex
     - elementwise

   * - :ref:`minimum(x, y) <minimum>`
     - :math:`\min \left\{x, y\right\}`
     - :math:`x, y \in \mathbf{R}`
     - sign depends |_| on |_| x,y

       |incr| incr.
     - |concave| concave
     - elementwise

   * - :ref:`multiply(c, x) <multiply>`

       :math:`c \in \mathbf{R}`
     - c*x
     - :math:`x \in\mathbf{R}`
     - :math:`\mathrm{sign}(cx)`

       monotonicity depends |_| on |_| c
     - |affine| affine
     - elementwise

   * - :ref:`neg(x) <neg>`
     - :math:`\max \left\{-x, 0 \right\}`
     - :math:`x \in \mathbf{R}`
     - |positive| positive

       |decr| decr.
     - |convex| convex
     - elementwise

   * - :ref:`pos(x) <pos>`
     - :math:`\max \left\{x, 0 \right\}`
     - :math:`x \in \mathbf{R}`
     - |positive| positive

       |incr| incr.
     - |convex| convex
     - elementwise

   * - :ref:`power(x, 0) <power>`
     - :math:`1`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - constant
     - elementwise

   * - :ref:`power(x, 1) <power>`
     - :math:`x`
     - :math:`x \in \mathbf{R}`
     - same sign as x
     
       |incr| incr.
     - |affine| affine
     - elementwise

   * - :ref:`power(x, p) <power>`

       :math:`p = 2, 4, 8, \ldots`
     - :math:`x^p`
     - :math:`x \in \mathbf{R}`
     - |positive| positive

       |incr| for :math:`x \geq 0`
       |decr| for :math:`x \leq 0`
     - |convex| convex
     - elementwise

   * - :ref:`power(x, p) <power>`

       :math:`p < 0`
     - :math:`x^p`
     - :math:`x > 0`
     - |positive| positive

       |decr| decr.
     - |convex| convex
     - elementwise

   * - :ref:`power(x, p) <power>`

       :math:`0 < p < 1`
     - :math:`x^p`
     - :math:`x \geq 0`
     - |positive| positive
       
       |incr| incr.
     - |concave| concave
     - elementwise

   * - :ref:`power(x, p) <power>`

       :math:`p > 1,\ p \neq 2, 4, 8, \ldots`

     - :math:`x^p`
     - :math:`x \geq 0`
     - |positive| positive

       |incr| incr.
     - |convex| convex
     - elementwise

   * - :ref:`real(x) <real-atom>`

     - real part of a complex number
     - :math:`x \in \mathbf{C}`
     - |unknown| unknown sign

       |incr| incr.
     - |affine| affine
     - elementwise

   * - :ref:`rel_entr(x, y) <rel-entr>`

     - :math:`x \log(x/y)`
     - :math:`x > 0`

       :math:`y > 0`
     - |unknown| unknown sign
       
       |decr| in :math:`y`
     - |convex| convex
     - elementwise

   * - :ref:`scalene(x, alpha, beta) <scalene>`

       :math:`\text{alpha} \geq 0`

       :math:`\text{beta} \geq 0`
     - :math:`\alpha\mathrm{pos}(x)+ \beta\mathrm{neg}(x)`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
       
       |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`
     - |convex| convex
     - elementwise

   * - :ref:`sqrt(x) <sqrt>`

     - :math:`\sqrt x`
     - :math:`x \geq 0`
     - |positive| positive

       |incr| incr.
     - |concave| concave
     - elementwise

   * - :ref:`square(x) <square>`

     - :math:`x^2`
     - :math:`x \in \mathbf{R}`
     - |positive| positive

       |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`
     - |convex| convex
     - elementwise

   * - :ref:`xexp(x) <xexp>`

     - :math:`x e^x`
     - :math:`x \geq 0`
     - |positive| positive

       |incr| incr.
     - |convex| convex
     - elementwise

   * - :ref:`bmat() <bmat>`

     - :math:`\left[\begin{matrix} X^{(1,1)} & .. &  X^{(1,q)} \\ \vdots &   & \vdots \\ X^{(p,1)} & .. &   X^{(p,q)} \end{matrix}\right]`
     - :math:`X^{(i,j)} \in\mathbf{R}^{m_i \times n_j}`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`convolve(c, x) <convolve>`

       :math:`c\in\mathbf{R}^m`
     - :math:`c*x`
     - :math:`x\in \mathbf{R}^n`
     - monotonicity depends |_| on |_| c
     - |affine| affine
     - matrix

   * - :ref:`cumsum(X, axis=0) <cumsum>`

     - cumulative sum along given axis.
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`diag(x) <diag>`

     - :math:`\left[\begin{matrix}x_1  & &  \\& \ddots & \\& & x_n\end{matrix}\right]`
     - :math:`x \in\mathbf{R}^{n}`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`diag(X) <diag>`
     - :math:`\left[\begin{matrix}X_{11}  \\\vdots \\X_{nn}\end{matrix}\right]`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`diff(X, k=1, axis=0) <diff>`

       :math:`k \in 0,1,2,\ldots`
     - kth order differences along given axis
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`hstack([X1, ..., Xk]) <hstack>`

     - :math:`\left[\begin{matrix}X^{(1)}  \cdots    X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m \times n_i}`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`kron(X, Y) <kron>`

       constant :math:`X\in\mathbf{R}^{p \times q}`
     - :math:`\left[\begin{matrix}X_{11}Y & .. & X_{1q}Y \\ \vdots  &        & \vdots \\ X_{p1}Y & .. & X_{pq}Y     \end{matrix}\right]`
     - :math:`Y \in \mathbf{R}^{m \times n}`
     - monotonicity depends on :math:`X`
     - |affine| affine
     - matrix

   * - :ref:`kron(X, Y) <kron>`

       constant :math:`Y\in\mathbf{R}^{m \times n}`
     - :math:`\left[\begin{matrix}X_{11}Y & .. & X_{1q}Y \\ \vdots  &        & \vdots \\ X_{p1}Y & .. & X_{pq}Y     \end{matrix}\right]`
     - :math:`X \in \mathbf{R}^{p \times q}`
     - monotonicity depends on :math:`Y`
     - |affine| affine
     - matrix

   * - :ref:`outer(x, y) <outer>`

       constant :math:`y \in \mathbf{R}^m`
     - :math:`x y^T`
     - :math:`x \in \mathbf{R}^n`
     - monotonicity depends on :math:`Y`
     - |affine| affine
     - matrix

   * - :ref:`partial_trace(X, dims, axis=0) <ptrace>`

     - partial trace
     - :math:`X \in\mathbf{R}^{n \times n}`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`partial_transpose(X, dims, axis=0) <ptrans>`

     - partial transpose
     - :math:`X \in\mathbf{R}^{n \times n}`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`reshape(X, (m', n'), order='F') <reshape>`

     - :math:`X' \in\mathbf{R}^{m' \times n'}`
     - :math:`X \in\mathbf{R}^{m \times n}`

       :math:`m'n' = mn`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`upper_tri(X) <upper_tri>`

     - flatten the strictly upper-triangular part of :math:`X`
     - :math:`X \in \mathbf{R}^{n \times n}`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`vec(X) <vec>`

     - :math:`x' \in\mathbf{R}^{mn}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`vec_to_upper_tri(X, strict=False) <vec_to_upper_tri>`

     - :math:`x' \in\mathbf{R}^{n(n-1)/2}` for ``strict=True``

       :math:`x' \in\mathbf{R}^{n(n+1)/2}` for ``strict=False``
     - :math:`X \in\mathbf{R}^{n \times n}`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`vstack([X1, ..., Xk]) <vstack>`

     - :math:`\left[\begin{matrix}X^{(1)}  \\ \vdots  \\X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m_i \times n}`
     - |incr| incr.
     - |affine| affine
     - matrix

   * - :ref:`geo_mean(x) <geo-mean>`

       :ref:`geo_mean(x, p) <geo-mean>`

       :math:`p \in \mathbf{R}^n_{+}`

       :math:`p \neq 0`
     - :math:`x_1^{1/n} \cdots x_n^{1/n}`

       :math:`\left(x_1^{p_1} \cdots x_n^{p_n}\right)^{\frac{1}{\mathbf{1}^T p}}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - |incr| incr.
     - |affine| log-log affine
     - scalar

   * - :ref:`harmonic_mean(x) <harmonic-mean>`
     - :math:`\frac{n}{\frac{1}{x_1} + \cdots + \frac{1}{x_n}}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - |incr| incr.
     - |concave| log-log concave
     - scalar

   * - :ref:`max(X) <max>`

     - :math:`\max_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - scalar

   * - :ref:`min(X) <min>`

     - :math:`\min_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |incr| incr.
     - |concave| log-log concave
     - scalar

   * - :ref:`norm(x) <norm>`

       norm(x, 2)

     - :math:`\sqrt{\sum_{i} \lvert x_{i} \rvert^2 }`
     - :math:`X \in\mathbf{R}^{n}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - scalar

   * - :ref:`norm(X, "fro") <norm>`
     - :math:`\sqrt{\sum_{ij}X_{ij}^2 }`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - scalar

   * - :ref:`norm(X, 1) <norm>`
     - :math:`\sum_{ij}\lvert X_{ij} \rvert`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - scalar

   * - :ref:`norm(X, "inf") <norm>`
     - :math:`\max_{ij} \{\lvert X_{ij} \rvert\}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - scalar

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`p \geq 1`

       or ``p = 'inf'``
     - :math:`\left(\sum_{ij} |X_{ij}|^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - scalar

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`0 < p < 1`
     - :math:`\left(\sum_{ij} X_{ij}^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - scalar

   * - :ref:`prod(X) <prod>`

     - :math:`\prod_{ij}X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |incr| incr.
     - |affine| log-log affine
     - scalar

   * - :ref:`quad_form(x, P) <quad-form>`
     - :math:`x^T P x`
     - :math:`x \in \mathbf{R}^n`, :math:`P \in \mathbf{R}^{n \times n}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - scalar

   * - :ref:`quad_over_lin(X, y) <quad-over-lin>`
     - :math:`\left(\sum_{ij}X_{ij}^2\right)/y`
     - :math:`x \in \mathbf{R}^n_{++}`

       :math:`y > 0`
     - |incr| in :math:`X_{ij}`

       |decr| decr. in :math:`y`
     - |convex| log-log convex
     - scalar

   * - :ref:`sum(X) <sum>`

     - :math:`\sum_{ij}X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - scalar

   * - :ref:`sum_squares(X) <sum-squares>`

     - :math:`\sum_{ij}X_{ij}^2`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - scalar

   * - :ref:`trace(X) <trace>`

     - :math:`\mathrm{tr}\left(X \right)`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - scalar

   * - :ref:`pf_eigenvalue(X) <pf-eigenvalue>`

     - spectral radius of :math:`X`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - scalar

   * - :ref:`diff_pos(x, y) <diff_pos>`
     - :math:`x - y`
     - :math:`0 < y < x`
     - |incr| incr.  in :math:`x`

       |decr| decr. in :math:`y`
     - |concave| log-log concave
     - elementwise

   * - :ref:`entr(x) <entr>`

     - :math:`-x \log (x)`
     - :math:`0 < x < 1`
     - None
     - |concave| log-log concave
     - elementwise

   * - :ref:`exp(x) <exp>`

     - :math:`e^x`
     - :math:`x > 0`
     - |incr| incr.
     - |convex| log-log convex
     - elementwise

   * - :ref:`log(x) <log>`

     - :math:`\log(x)`
     - :math:`x > 1`
     - |incr| incr.
     - |concave| log-log concave
     - elementwise

   * - :ref:`maximum(x, y) <maximum>`

     - :math:`\max \left\{x, y\right\}`
     - :math:`x,y > 0`
     - |incr| incr.
     - |convex| log-log convex
     - elementwise

   * - :ref:`minimum(x, y) <minimum>`
     - :math:`\min \left\{x, y\right\}`
     - :math:`x, y > 0`
     - |incr| incr.
     - |concave| log-log concave
     - elementwise

   * - :ref:`multiply(x, y) <multiply>`
     - :math:`x*y`
     - :math:`x, y > 0`
     - |incr| incr.
     - |affine| log-log affine
     - elementwise

   * - :ref:`one_minus_pos(x) <one-minus-pos>`
     - :math:`1 - x`
     - :math:`0 < x < 1`
     - |decr| decr.
     - |concave| log-log concave
     - elementwise

   * - :ref:`power(x, 0) <power>`
     - :math:`1`
     - :math:`x > 0`
     - constant
     - constant
     - elementwise

   * - :ref:`power(x, p) <power>`
     - :math:`x`
     - :math:`x > 0`
     - |incr| for :math:`p > 0`

       |decr| for :math:`p < 0`
     - |affine| log-log affine
     - elementwise

   * - :ref:`sqrt(x) <sqrt>`


     - :math:`\sqrt x`
     - :math:`x > 0`
     - |incr| incr.
     - |affine| log-log affine
     - elementwise

   * - :ref:`square(x) <square>`

     - :math:`x^2`
     - :math:`x > 0`
     - |incr| incr.
     - |affine| log-log affine
     - elementwise

   * - :ref:`xexp(x) <xexp>`

     - :math:`x e^x`
     - :math:`x > 0`
     - |incr| incr.
     - |convex| log-log convex
     - elementwise

   * - :ref:`bmat() <bmat>`

     - :math:`\left[\begin{matrix} X^{(1,1)} & .. &  X^{(1,q)} \\ \vdots &   & \vdots \\ X^{(p,1)} & .. &   X^{(p,q)} \end{matrix}\right]`
     - :math:`X^{(i,j)} \in\mathbf{R}^{m_i \times n_j}_{++}`
     - |incr| incr.
     - |affine| log-log affine
     - matrix

   * - :ref:`diag(x) <diag>`

     - :math:`\left[\begin{matrix}x_1  & &  \\& \ddots & \\& & x_n\end{matrix}\right]`
     - :math:`x \in\mathbf{R}^{n}_{++}`
     - |incr| incr.
     - |affine| log-log affine
     - matrix

   * - :ref:`diag(X) <diag>`
     - :math:`\left[\begin{matrix}X_{11}  \\\vdots \\X_{nn}\end{matrix}\right]`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}`
     - |incr| incr.
     - |affine| log-log affine
     - matrix

   * - :ref:`eye_minus_inv(X) <eye_minus_inv>`
     - :math:`(I - X)^{-1}`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}`,
     
       :math:`\lambda_{\text{pf}}(X) < 1`
     - |incr| incr.
     - |convex| log-log convex
     - matrix

   * - :ref:`gmatmul(A, x) <gmatmul>`

       :math:`A \in \mathbf{R}^{m \times n}`
     - :math:`\left[\begin{matrix}\prod_{j=1}^n x_j^{A_{1j}} \\\vdots \\\prod_{j=1}^n x_j^{A_{mj}}\end{matrix}\right]`
     - :math:`x \in \mathbf{R}^n_{++}`
     - |incr| for :math:`A_{ij} \geq 0`

       |decr| for :math:`A_{ij} \leq 0`
     - |affine| log-log affine
     - matrix

   * - :ref:`hstack([X1, ..., Xk]) <hstack>`
     - :math:`\left[\begin{matrix}X^{(1)}  \cdots    X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m \times n_i}_{++}`
     - |incr| incr.
     - |affine| log-log affine
     - matrix

   * - :ref:`matmul(X, Y) <matmul>`
     - :math:`XY`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
       :math:`Y \in\mathbf{R}^{n \times p}_{++}`
     - |incr| incr.
     - |convex| log-log convex
     - matrix

   * - :ref:`resolvent(X) <resolvent>`
     - :math:`(sI - X)^{-1}`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}`
     
       :math:`\lambda_{\text{pf}}(X) < s`
     - |incr| incr.
     - |convex| log-log convex
     - matrix

   * - :ref:`reshape(X, (m', n')) <reshape>`

     - :math:`X' \in\mathbf{R}^{m' \times n'}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`

       :math:`m'n' = mn`
     - |incr| incr.
     - |affine| log-log affine
     - matrix

   * - :ref:`vec(X) <vec>`

     - :math:`x' \in\mathbf{R}^{mn}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - |incr| incr.
     - |affine| log-log affine
     - matrix

   * - :ref:`vstack([X1, ..., Xk]) <vstack>`

     - :math:`\left[\begin{matrix}X^{(1)}  \\ \vdots  \\X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m_i \times n}_{++}`
     - |incr| incr.
     - |affine| log-log affine
     - matrix

.. |positive| image:: /tutorial/functions/functions_files/positive.svg
              :width: 15px
              :height: 15px

.. |negative| image:: /tutorial/functions/functions_files/negative.svg
              :width: 15px
              :height: 15px

.. |unknown| image:: /tutorial/functions/functions_files/unknown.svg
              :width: 15px
              :height: 15px

.. |convex| image:: /tutorial/functions/functions_files/convex.svg
              :width: 15px
              :height: 15px

.. |concave| image:: /tutorial/functions/functions_files/concave.svg
              :width: 15px
              :height: 15px

.. |affine| image:: /tutorial/functions/functions_files/affine.svg
              :width: 15px
              :height: 15px

.. |incr| image:: /tutorial/functions/functions_files/increasing.svg
              :width: 15px
              :height: 15px

.. |decr| image:: /tutorial/functions/functions_files/decreasing.svg
              :width: 15px
              :height: 15px