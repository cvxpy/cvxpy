.. |_| unicode:: 0xA0
   :trim:

.. list-table::
   :class: atomic-functions
   :header-rows: 1

   * - Function
     - Meaning
     - Domain
     - Sign
     - Curvature |_|
     - Monotonicity
     - Type

   * - :ref:`dotsort(X,W) <dotsort>`

       constant :math:`W \in \mathbf{R}^{o \times p}`
     - :math:`\langle sort\left(vec(X)\right), sort\left(vec(W)\right) \rangle`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - depends on :math:`X`, :math:`W`
     - |convex| convex
     - |incr| for :math:`\min(W) \geq 0`
     - scalar

   * - :ref:`geo_mean(x) <geo-mean>`

       :ref:`geo_mean(x, p) <geo-mean>`

       :math:`p \in \mathbf{R}^n_{+}`

       :math:`p \neq 0`
     - :math:`x_1^{1/n} \cdots x_n^{1/n}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - |positive| positive
     - |concave| concave
     - |incr| incr.
     - scalar

   * - :ref:`harmonic_mean(x) <harmonic-mean>`
     - :math:`\frac{n}{\frac{1}{x_1} + \cdots + \frac{1}{x_n}}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - |positive| positive
     - |concave| concave
     - |incr| incr.
     - scalar

   * - :ref:`inv_prod(x) <inv-prod>`
     - :math:`(x_1\cdots x_n)^{-1}`
     - :math:`x \in \mathbf{R}^n_+`
     - |positive| positive
     - |convex| convex
     - |decr| decr.
     - scalar

   * - :ref:`lambda_max(X) <lambda-max>`
     - :math:`\lambda_{\max}(X)`
     - :math:`X \in \mathbf{S}^n`
     - |unknown| unknown
     - |convex| convex
     - None
     - scalar

   * - :ref:`lambda_min(X) <lambda-min>`
     - :math:`\lambda_{\min}(X)`
     - :math:`X \in \mathbf{S}^n`
     - |unknown| unknown
     - |concave| concave
     - None
     - scalar

   * - :ref:`lambda_sum_largest(X,k) <lambda-sum-largest>`

       :math:`k = 1,\ldots, n`
     - :math:`\text{sum of $k$ largest}\\ \text{eigenvalues of $X$}`
     - :math:`X \in\mathbf{S}^{n}`
     - |unknown| unknown
     - |convex| convex
     - None
     - scalar

   * - :ref:`lambda_sum_smallest(X,k) <lambda-sum-smallest>`

       :math:`k = 1,\ldots, n`
     - :math:`\text{sum of $k$ smallest}\\ \text{eigenvalues of $X$}`
     - :math:`X \in\mathbf{S}^{n}`
     - |unknown| unknown
     - |concave| concave
     - None
     - scalar

   * - :ref:`log_det(X) <log-det>`

     - :math:`\log \left(\det (X)\right)`
     - :math:`X \in \mathbf{S}^n_+`
     - |unknown| unknown
     - |concave| concave
     - None
     - scalar

   * - :ref:`log_sum_exp(X) <log-sum-exp>`

     - :math:`\log \left(\sum_{ij}e^{X_{ij}}\right)`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |unknown| unknown
     - |convex| convex
     - |incr| incr.
     - scalar

   * - :ref:`matrix_frac(x, P) <matrix-frac>`

     - :math:`x^T P^{-1} x`
     - :math:`x \in \mathbf{R}^n`
     - |positive| positive
     - |convex| convex
     - None
     - scalar

   * - :ref:`max(X) <max>`

     - :math:`\max_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |convex| convex
     - |incr| incr.
     - scalar

   * - :ref:`mean(X) <mean>`

     - :math:`\frac{1}{m n}\sum_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |affine| affine
     - |incr| incr.
     - scalar

   * - :ref:`min(X) <min>`

     - :math:`\min_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |concave| concave
     - |incr| incr.
     - scalar

   * - :ref:`mixed_norm(X, p, q) <mixed-norm>`

     - :math:`\left(\sum_k\left(\sum_l\lvert x_{k,l}\rvert^p\right)^{q/p}\right)^{1/q}`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - |positive| positive
     - |convex| convex
     - None
     - scalar

   * - :ref:`norm(x) <norm>`

       norm(x, 2)
     - :math:`\sqrt{\sum_{i} \lvert x_{i} \rvert^2 }`
     - :math:`X \in\mathbf{R}^{n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x_{i} \geq 0`
     - scalar

   * - :ref:`norm(x, 1) <norm>`
     - :math:`\sum_{i}\lvert x_{i} \rvert`
     - :math:`x \in\mathbf{R}^{n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x_{i} \geq 0`
     - scalar

   * - :ref:`norm(x, "inf") <norm>`
     - :math:`\max_{i} \{\lvert x_{i} \rvert\}`
     - :math:`x \in\mathbf{R}^{n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x_{i} \geq 0`
     - scalar

   * - :ref:`norm(X, "fro") <norm>`
     - :math:`\sqrt{\sum_{ij}X_{ij}^2 }`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`X_{ij} \geq 0`
     - scalar

   * - :ref:`norm(X, 1) <norm>`
     - :math:`\max_{j} \|X_{:,j}\|_1`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`X_{ij} \geq 0`
     - scalar

   * - :ref:`norm(X, "inf") <norm>`
     - :math:`\max_{i} \|X_{i,:}\|_1`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`X_{ij} \geq 0`
     - scalar

   * - :ref:`norm(X, "nuc") <norm>`
     - :math:`\mathrm{tr}\left(\left(X^T X\right)^{1/2}\right)`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - None
     - scalar

   * - :ref:`norm(X) <norm>`
       norm(X, 2)
     - :math:`\sqrt{\lambda_{\max}\left(X^T X\right)}`
     - :math:`X in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - None
     - scalar

   * - :ref:`perspective(f(x),s) <perspective>`

     - :math:`sf(x/s)`
     - :math:`x \in \mathop{\bf dom} f`
       :math:`s \geq 0`
     - same as f
     - |convex| / |concave|
       same as :math:`f`
     - None
     - scalar

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`p \geq 1`
       or ``p = 'inf'``
     - :math:`\|X\|_p = \left(\sum_{ij} |X_{ij}|^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`X_{ij} \geq 0`
     - scalar

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`p < 1`, :math:`p \neq 0`
     - :math:`\|X\|_p = \left(\sum_{ij} X_{ij}^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}_+`
     - |positive| positive
     - |concave| concave
     - |incr| incr.
     - scalar

   * - :ref:`ptp(X) <ptp>`

     - :math:`\max_{ij} X_{ij} - \min_{ij} X_{ij}`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - None
     - scalar

   * - :ref:`quad_form(x, P) <quad-form>`

       constant :math:`P \in \mathbf{S}^n_+`
     - :math:`x^T P x`
     - :math:`x \in \mathbf{R}^n`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x_i \geq 0`
     - scalar

   * - :ref:`quad_form(x, P) <quad-form>`

       constant :math:`P \in \mathbf{S}^n_-`
     - :math:`x^T P x`
     - :math:`x \in \mathbf{R}^n`
     - |negative| negative
     - |concave| concave
     - |decr| for :math:`x_i \geq 0`
     - scalar

   * - :ref:`quad_form(c, X) <quad-form>`

       constant :math:`c \in \mathbf{R}^n`
     - :math:`c^T X c`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - depends |_| on |_| c, |_| X
     - |affine| affine
     - depends |_| on |_| c
     - scalar

   * - :ref:`quad_over_lin(X, y) <quad-over-lin>`

     - :math:`\left(\sum_{ij}X_{ij}^2\right)/y`
     - :math:`x \in \mathbf{R}^n`
       :math:`y > 0`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`X_{ij} \geq 0`
       |decr| for :math:`X_{ij} \leq 0`
       |decr| decr. in :math:`y`
     - scalar

   * - :ref:`std(X) <std>`

     - :math:`\sqrt{\frac{1}{mn} \sum_{ij}\left(X_{ij} - \frac{1}{mn}\sum_{k\ell} X_{k\ell}\right)^2}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - None
     - scalar

   * - :ref:`sum(X) <sum>`

     - :math:`\sum_{ij}X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |affine| affine
     - |incr| incr.
     - scalar

   * - :ref:`sum_largest(X, k) <sum-largest>`

       :math:`k = 1,2,\ldots`
     - :math:`\text{sum of } k\text{ largest }X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |convex| convex
     - |incr| incr.
     - scalar

   * - :ref:`sum_smallest(X, k) <sum-smallest>`

       :math:`k = 1,2,\ldots`
     - :math:`\text{sum of } k\text{ smallest }X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - same as X
     - |concave| concave
     - |incr| incr.
     - scalar

   * - :ref:`sum_squares(X) <sum-squares>`

     - :math:`\sum_{ij}X_{ij}^2`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`X_{ij} \geq 0`
       |decr| for :math:`X_{ij} \leq 0`
     - scalar

   * - :ref:`trace(X) <trace>`

     - :math:`\mathrm{tr}\left(X \right)`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - same as X
     - |affine| affine
     - |incr| incr.
     - scalar

   * - :ref:`tr_inv(X) <tr_inv>`

     - :math:`\mathrm{tr}\left(X^{-1} \right)`
     - :math:`X \in\mathbf{S}^n_{++}`
     - |positive| positive
     - |convex| convex
     - None
     - scalar

   * - :ref:`tv(x) <tv>`

     - :math:`\sum_{i}|x_{i+1} - x_i|`
     - :math:`x \in \mathbf{R}^n`
     - |positive| positive
     - |convex| convex
     - None
     - scalar

   * - :ref:`tv(X) <tv>`
     - :math:`\sum_{ij}\left\| \left[\begin{matrix} X_{i+1,j} - X_{ij} \\ X_{i,j+1} -X_{ij} \end{matrix}\right] \right\|_2`
     - :math:`X \in \mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - None
     - scalar

   * - :ref:`tv([X1,...,Xk]) <tv>`
     - :math:`\sum_{ij}\left\| \left[\begin{matrix} X_{i+1,j}^{(1)} - X_{ij}^{(1)} \\ X_{i,j+1}^{(1)} -X_{ij}^{(1)} \\ \vdots \\ X_{i+1,j}^{(k)} - X_{ij}^{(k)} \\ X_{i,j+1}^{(k)} -X_{ij}^{(k)}  \end{matrix}\right] \right\|_2`
     - :math:`X^{(i)} \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - None
     - scalar

   * - :ref:`var(X) <var>`

     - :math:`{\frac{1}{mn} \sum_{ij}\left(X_{ij} - \frac{1}{mn}\sum_{k\ell} X_{k\ell}\right)^2}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - |positive| positive
     - |convex| convex
     - None
     - scalar

   * - :ref:`abs(x) <abs>`

     - :math:`\lvert x \rvert`
     - :math:`x \in \mathbf{C}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x \geq 0`
     - elementwise

   * - :ref:`conj(x) <conj>`

     - complex conjugate
     - :math:`x \in \mathbf{C}`
     - |unknown| unknown
     - |affine| affine
     - None
     - elementwise

   * - :ref:`entr(x) <entr>`

     - :math:`-x \log (x)`
     - :math:`x > 0`
     - |unknown| unknown
     - |concave| concave
     - None
     - elementwise

   * - :ref:`exp(x) <exp>`

     - :math:`e^x`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| incr.
     - elementwise

   * - :ref:`huber(x, M=1) <huber>`

       :math:`M \geq 0`
     - :math:`\begin{cases}x^2 &|x| \leq M  \\2M|x| - M^2&|x| >M\end{cases}`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x \geq 0`
       |decr| for :math:`x \leq 0`
     - elementwise

   * - :ref:`imag(x) <imag-atom>`

     - imaginary part of a complex number
     - :math:`x \in \mathbf{C}`
     - |unknown| unknown
     - |affine| affine
     - none
     - elementwise

   * - :ref:`inv_pos(x) <inv-pos>`

     - :math:`1/x`
     - :math:`x > 0`
     - |positive| positive
     - |convex| convex
     - |decr| decr.
     - elementwise

   * - :ref:`kl_div(x, y) <kl-div>`

     - :math:`x \log(x/y) - x + y`
     - :math:`x > 0`
       :math:`y > 0`
     - |positive| positive
     - |convex| convex
     - None
     - elementwise

   * - :ref:`log(x) <log>`

     - :math:`\log(x)`
     - :math:`x > 0`
     - |unknown| unknown
     - |concave| concave
     - |incr| incr.
     - elementwise

   * - :ref:`log_normcdf(x) <log-normcdf>`

     - :ref:`approximate <clarifyelementwise>` log of the standard normal CDF
     - :math:`x \in \mathbf{R}`
     - |negative| negative
     - |concave| concave
     - |incr| incr.
     - elementwise

   * - :ref:`log1p(x) <log1p>`

     - :math:`\log(x+1)`
     - :math:`x > -1`
     - same as x
     - |concave| concave
     - |incr| incr.
     - elementwise

   * - :ref:`loggamma(x) <loggamma>`

     - :ref:`approximate <clarifyelementwise>` `log of the Gamma function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loggamma.html>`_
     - :math:`x > 0`
     - |unknown| unknown
     - |convex| convex
     - None
     - elementwise

   * - :ref:`logistic(x) <logistic>`

     - :math:`\log(1 + e^{x})`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| incr.
     - elementwise

   * - :ref:`maximum(x, y) <maximum>`

     - :math:`\max \left\{x, y\right\}`
     - :math:`x,y \in \mathbf{R}`
     - depends on x,y
     - |convex| convex
     - |incr| incr.
     - elementwise

   * - :ref:`minimum(x, y) <minimum>`
     - :math:`\min \left\{x, y\right\}`
     - :math:`x, y \in \mathbf{R}`
     - depends |_| on |_| x,y
     - |concave| concave
     - |incr| incr.
     - elementwise

   * - :ref:`multiply(c, x) <multiply>`

       :math:`c \in \mathbf{R}`
     - c*x
     - :math:`x \in\mathbf{R}`
     - :math:`\mathrm{sign}(cx)`
     - |affine| affine
     - depends |_| on |_| c
     - elementwise

   * - :ref:`neg(x) <neg>`
     - :math:`\max \left\{-x, 0 \right\}`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |decr| decr.
     - elementwise

   * - :ref:`pos(x) <pos>`
     - :math:`\max \left\{x, 0 \right\}`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| incr.
     - elementwise

   * - :ref:`power(x, 0) <power>`
     - :math:`1`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - constant
     - |_|
     - elementwise

   * - :ref:`power(x, 1) <power>`
     - :math:`x`
     - :math:`x \in \mathbf{R}`
     - same as x
     - |affine| affine
     - |incr| incr.
     - elementwise

   * - :ref:`power(x, p) <power>`

       :math:`p = 2, 4, 8, \ldots`
     - :math:`x^p`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x \geq 0`
       |decr| for :math:`x \leq 0`
     - elementwise

   * - :ref:`power(x, p) <power>`

       :math:`p < 0`
     - :math:`x^p`
     - :math:`x > 0`
     - |positive| positive
     - |convex| convex
     - |decr| decr.
     - elementwise

   * - :ref:`power(x, p) <power>`

       :math:`0 < p < 1`
     - :math:`x^p`
     - :math:`x \geq 0`
     - |positive| positive
     - |concave| concave
     - |incr| incr.
     - elementwise

   * - :ref:`power(x, p) <power>`

       :math:`p > 1,\ p \neq 2, 4, 8, \ldots`

     - :math:`x^p`
     - :math:`x \geq 0`
     - |positive| positive
     - |convex| convex
     - |incr| incr.
     - elementwise

   * - :ref:`real(x) <real-atom>`

     - real part of a complex number
     - :math:`x \in \mathbf{C}`
     - |unknown| unknown
     - |affine| affine
     - |incr| incr.
     - elementwise

   * - :ref:`rel_entr(x, y) <rel-entr>`

     - :math:`x \log(x/y)`
     - :math:`x > 0`

       :math:`y > 0`
     - |unknown| unknown
     - |convex| convex
     - None in :math:`x`

       |decr| in :math:`y`
     - elementwise

   * - :ref:`scalene(x, alpha, beta) <scalene>`

       :math:`\text{alpha} \geq 0`

       :math:`\text{beta} \geq 0`
     - :math:`\alpha\mathrm{pos}(x)+ \beta\mathrm{neg}(x)`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`
     - elementwise

   * - :ref:`sqrt(x) <sqrt>`

     - :math:`\sqrt x`
     - :math:`x \geq 0`
     - |positive| positive
     - |concave| concave
     - |incr| incr.
     - elementwise

   * - :ref:`square(x) <square>`

     - :math:`x^2`
     - :math:`x \in \mathbf{R}`
     - |positive| positive
     - |convex| convex
     - |incr| for :math:`x \geq 0`

       |decr| for :math:`x \leq 0`
     - elementwise

   * - :ref:`xexp(x) <xexp>`

     - :math:`x e^x`
     - :math:`x \geq 0`
     - |positive| positive
     - |convex| convex
     - |incr| incr.
     - elementwise

   * - :ref:`bmat([[X11,...,X1q],
       ...,
       [Xp1,...,Xpq]]) <bmat>`

     - :math:`\left[\begin{matrix} X^{(1,1)} &  \cdots &  X^{(1,q)} \\ \vdots &   & \vdots \\ X^{(p,1)} & \cdots &   X^{(p,q)} \end{matrix}\right]`
     - :math:`X^{(i,j)} \in\mathbf{R}^{m_i \times n_j}`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`convolve(c, x) <convolve>`

       :math:`c\in\mathbf{R}^m`
     - :math:`c*x`
     - :math:`x\in \mathbf{R}^n`
     - none
     - |affine| affine
     - depends |_| on |_| c
     - matrix

   * - :ref:`cumsum(X, axis=0) <cumsum>`

     - cumulative sum along given axis.
     - :math:`X \in \mathbf{R}^{m \times n}`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`diag(x) <diag>`

     - :math:`\left[\begin{matrix}x_1  & &  \\& \ddots & \\& & x_n\end{matrix}\right]`
     - :math:`x \in\mathbf{R}^{n}`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`diag(X) <diag>`
     - :math:`\left[\begin{matrix}X_{11}  \\\vdots \\X_{nn}\end{matrix}\right]`
     - :math:`X \in\mathbf{R}^{n \times n}`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`diff(X, k=1, axis=0) <diff>`

       :math:`k \in 0,1,2,\ldots`
     - kth order differences along given axis
     - :math:`X \in\mathbf{R}^{m \times n}`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`hstack([X1, ..., Xk]) <hstack>`

     - :math:`\left[\begin{matrix}X^{(1)}  \cdots    X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m \times n_i}`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`kron(X, Y) <kron>`

       constant :math:`X\in\mathbf{R}^{p \times q}`
     - :math:`\left[\begin{matrix}X_{11}Y & \cdots & X_{1q}Y \\ \vdots  &        & \vdots \\ X_{p1}Y &  \cdots      & X_{pq}Y     \end{matrix}\right]`
     - :math:`Y \in \mathbf{R}^{m \times n}`
     - none
     - |affine| affine
     - depends on :math:`X`
     - matrix

   * - :ref:`kron(X, Y) <kron>`

       constant :math:`Y\in\mathbf{R}^{m \times n}`
     - :math:`\left[\begin{matrix}X_{11}Y & \cdots & X_{1q}Y \\ \vdots  &        & \vdots \\ X_{p1}Y &  \cdots      & X_{pq}Y     \end{matrix}\right]`
     - :math:`X \in \mathbf{R}^{p \times q}`
     - none
     - |affine| affine
     - depends on :math:`Y`
     - matrix

   * - :ref:`outer(x, y) <outer>`

       constant :math:`y \in \mathbf{R}^m`
     - :math:`x y^T`
     - :math:`x \in \mathbf{R}^n`
     - none
     - |affine| affine
     - depends on :math:`y`
     - matrix

   * - :ref:`partial_trace(X, dims, axis=0) <ptrace>`

     - partial trace
     - :math:`X \in\mathbf{R}^{n \times n}`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`partial_transpose(X, dims, axis=0) <ptrans>`

     - partial transpose
     - :math:`X \in\mathbf{R}^{n \times n}`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`reshape(X, (m', n'), order='F') <reshape>`

     - :math:`X' \in\mathbf{R}^{m' \times n'}`
     - :math:`X \in\mathbf{R}^{m \times n}`

       :math:`m'n' = mn`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`upper_tri(X) <upper_tri>`

     - flatten the strictly upper-triangular part of :math:`X`
     - :math:`X \in \mathbf{R}^{n \times n}`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`vec(X) <vec>`

     - :math:`x' \in\mathbf{R}^{mn}`
     - :math:`X \in\mathbf{R}^{m \times n}`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`vec_to_upper_tri(X, strict=False) <vec-to-upper-tri>`

     - :math:`x' \in\mathbf{R}^{n(n-1)/2}` for ``strict=True``

       :math:`x' \in\mathbf{R}^{n(n+1)/2}` for ``strict=False``
     - :math:`X \in\mathbf{R}^{n \times n}`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`vstack([X1, ..., Xk]) <vstack>`

     - :math:`\left[\begin{matrix}X^{(1)}  \\ \vdots  \\X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m_i \times n}`
     - none
     - |affine| affine
     - |incr| incr.
     - matrix

   * - :ref:`geo_mean(x) <geo-mean>`

       :ref:`geo_mean(x, p) <geo-mean>`

       :math:`p \in \mathbf{R}^n_{+}`

       :math:`p \neq 0`
     - :math:`x_1^{1/n} \cdots x_n^{1/n}`

       :math:`\left(x_1^{p_1} \cdots x_n^{p_n}\right)^{\frac{1}{\mathbf{1}^T p}}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - none
     - |affine| log-log affine
     - |incr| incr.
     - scalar

   * - :ref:`harmonic_mean(x) <harmonic-mean>`
     - :math:`\frac{n}{\frac{1}{x_1} + \cdots + \frac{1}{x_n}}`
     - :math:`x \in \mathbf{R}^n_{+}`
     - none
     - |concave| log-log concave
     - |incr| incr.
     - scalar

   * - :ref:`max(X) <max>`

     - :math:`\max_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - scalar

   * - :ref:`min(X) <min>`

     - :math:`\min_{ij}\left\{ X_{ij}\right\}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - none
     - |concave| log-log concave
     - |incr| incr.
     - scalar

   * - :ref:`norm(x) <norm>`

       norm(x, 2)

     - :math:`\sqrt{\sum_{i} \lvert x_{i} \rvert^2 }`
     - :math:`X \in\mathbf{R}^{n}_{++}`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - scalar

   * - :ref:`norm(X, "fro") <norm>`
     - :math:`\sqrt{\sum_{ij}X_{ij}^2 }`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - scalar

   * - :ref:`norm(X, 1) <norm>`
     - :math:`\sum_{ij}\lvert X_{ij} \rvert`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - scalar

   * - :ref:`norm(X, "inf") <norm>`
     - :math:`\max_{ij} \{\lvert X_{ij} \rvert\}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - scalar

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`p \geq 1`

       or ``p = 'inf'``
     - :math:`\|X\|_p = \left(\sum_{ij} |X_{ij}|^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}_{++}`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - scalar

   * - :ref:`pnorm(X, p) <pnorm_func>`

       :math:`0 < p < 1`
     - :math:`\|X\|_p = \left(\sum_{ij} X_{ij}^p \right)^{1/p}`
     - :math:`X \in \mathbf{R}^{m \times n}_{++}`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - scalar

   * - :ref:`prod(X) <prod>`

     - :math:`\prod_{ij}X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - none
     - |affine| log-log affine
     - |incr| incr.
     - scalar

   * - :ref:`quad_form(x, P) <quad-form>`
     - :math:`x^T P x`
     - :math:`x \in \mathbf{R}^n`, :math:`P \in \mathbf{R}^{n \times n}_{++}`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - scalar

   * - :ref:`quad_over_lin(X, y) <quad-over-lin>`
     - :math:`\left(\sum_{ij}X_{ij}^2\right)/y`
     - :math:`x \in \mathbf{R}^n_{++}`

       :math:`y > 0`
     - none
     - |convex| log-log convex
     - |incr| in :math:`X_{ij}`

       |decr| decr. in :math:`y`
     - scalar

   * - :ref:`sum(X) <sum>`

     - :math:`\sum_{ij}X_{ij}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - scalar

   * - :ref:`sum_squares(X) <sum-squares>`

     - :math:`\sum_{ij}X_{ij}^2`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - scalar

   * - :ref:`trace(X) <trace>`

     - :math:`\mathrm{tr}\left(X \right)`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - scalar

   * - :ref:`pf_eigenvalue(X) <pf-eigenvalue>`

     - spectral radius of :math:`X`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - scalar

   * - :ref:`diff_pos(x, y) <diff-pos>`
     - :math:`x - y`
     - :math:`0 < y < x`
     - none
     - |concave| log-log concave
     - |incr| incr.  in :math:`x`

       |decr| decr. in :math:`y`
     - elementwise

   * - :ref:`entr(x) <entr>`

     - :math:`-x \log (x)`
     - :math:`0 < x < 1`
     - none
     - |concave| log-log concave
     - None
     - elementwise

   * - :ref:`exp(x) <exp>`

     - :math:`e^x`
     - :math:`x > 0`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - elementwise

   * - :ref:`log(x) <log>`

     - :math:`\log(x)`
     - :math:`x > 1`
     - none
     - |concave| log-log concave
     - |incr| incr.
     - elementwise

   * - :ref:`maximum(x, y) <maximum>`

     - :math:`\max \left\{x, y\right\}`
     - :math:`x,y > 0`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - elementwise

   * - :ref:`minimum(x, y) <minimum>`
     - :math:`\min \left\{x, y\right\}`
     - :math:`x, y > 0`
     - none
     - |concave| log-log concave
     - |incr| incr.
     - elementwise

   * - :ref:`multiply(x, y) <multiply>`
     - :math:`x*y`
     - :math:`x, y > 0`
     - none
     - |affine| log-log affine
     - |incr| incr.
     - elementwise

   * - :ref:`one_minus_pos(x) <one-minus-pos>`
     - :math:`1 - x`
     - :math:`0 < x < 1`
     - none
     - |concave| log-log concave
     - |decr| decr.
     - elementwise

   * - :ref:`power(x, 0) <power>`
     - :math:`1`
     - :math:`x > 0`
     - none
     - constant
     - constant
     - elementwise

   * - :ref:`power(x, p) <power>`
     - :math:`x`
     - :math:`x > 0`
     - none
     - |affine| log-log affine
     - |incr| for :math:`p > 0`

       |decr| for :math:`p < 0`
     - elementwise

   * - :ref:`sqrt(x) <sqrt>`


     - :math:`\sqrt x`
     - :math:`x > 0`
     - none
     - |affine| log-log affine
     - |incr| incr.
     - elementwise

   * - :ref:`square(x) <square>`

     - :math:`x^2`
     - :math:`x > 0`
     - none
     - |affine| log-log affine
     - |incr| incr.
     - elementwise

   * - :ref:`xexp(x) <xexp>`

     - :math:`x e^x`
     - :math:`x > 0`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - elementwise

   * - :ref:`bmat([[X11,...,X1q],
       ...,
       [Xp1,...,Xpq]]) <bmat>`

     - :math:`\left[\begin{matrix} X^{(1,1)} &  \cdots &  X^{(1,q)} \\ \vdots &   & \vdots \\ X^{(p,1)} & \cdots &   X^{(p,q)} \end{matrix}\right]`
     - :math:`X^{(i,j)} \in\mathbf{R}^{m_i \times n_j}_{++}`
     - none
     - |affine| log-log affine
     - |incr| incr.
     - matrix

   * - :ref:`diag(x) <diag>`

     - :math:`\left[\begin{matrix}x_1  & &  \\& \ddots & \\& & x_n\end{matrix}\right]`
     - :math:`x \in\mathbf{R}^{n}_{++}`
     - none
     - |affine| log-log affine
     - |incr| incr.
     - matrix

   * - :ref:`diag(X) <diag>`
     - :math:`\left[\begin{matrix}X_{11}  \\\vdots \\X_{nn}\end{matrix}\right]`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}`
     - none
     - |affine| log-log affine
     - |incr| incr.
     - matrix

   * - :ref:`eye_minus_inv(X) <eye-minus-inv>`
     - :math:`(I - X)^{-1}`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}, \lambda_{\text{pf}}(X) < 1`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - matrix

   * - :ref:`gmatmul(A, x) <gmatmul>`

       :math:`A \in \mathbf{R}^{m \times n}`
     - :math:`\left[\begin{matrix}\prod_{j=1}^n x_j^{A_{1j}} \\\vdots \\\prod_{j=1}^n x_j^{A_{mj}}\end{matrix}\right]`
     - :math:`x \in \mathbf{R}^n_{++}`
     - none
     - |affine| log-log affine
     - |incr| for :math:`A_{ij} \geq 0`

       |decr| for :math:`A_{ij} \leq 0`
     - matrix

   * - :ref:`hstack([X1, ..., Xk]) <hstack>`
     - :math:`\left[\begin{matrix}X^{(1)}  \cdots    X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m \times n_i}_{++}`
     - none
     - |affine| log-log affine
     - |incr| incr.
     - matrix

   * - :ref:`matmul(X, Y) <matmul>`
     - :math:`XY`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}, Y \in\mathbf{R}^{n \times p}_{++}``
     - none
     - |convex| log-log convex
     - |incr| incr.
     - matrix

   * - :ref:`resolvent(X) <resolvent>`
     - :math:`(sI - X)^{-1}`
     - :math:`X \in\mathbf{R}^{n \times n}_{++}, \lambda_{\text{pf}}(X) < s`
     - none
     - |convex| log-log convex
     - |incr| incr.
     - matrix

   * - :ref:`reshape(X, (m', n')) <reshape>`

     - :math:`X' \in\mathbf{R}^{m' \times n'}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`

       :math:`m'n' = mn`
     - none
     - |affine| log-log affine
     - |incr| incr.
     - matrix

   * - :ref:`vec(X) <vec>`

     - :math:`x' \in\mathbf{R}^{mn}`
     - :math:`X \in\mathbf{R}^{m \times n}_{++}`
     - none
     - |affine| log-log affine
     - |incr| incr.
     - matrix

   * - :ref:`vstack([X1, ..., Xk]) <vstack>`

     - :math:`\left[\begin{matrix}X^{(1)}  \\ \vdots  \\X^{(k)}\end{matrix}\right]`
     - :math:`X^{(i)} \in\mathbf{R}^{m_i \times n}_{++}`
     - none
     - |affine| log-log affine
     - |incr| incr.
     - matrix

.. |positive| image:: functions_files/positive.svg
              :width: 15px
              :height: 15px

.. |negative| image:: functions_files/negative.svg
              :width: 15px
              :height: 15px

.. |unknown| image:: functions_files/unknown.svg
              :width: 15px
              :height: 15px

.. |convex| image:: functions_files/convex.svg
              :width: 15px
              :height: 15px

.. |concave| image:: functions_files/concave.svg
              :width: 15px
              :height: 15px

.. |affine| image:: functions_files/affine.svg
              :width: 15px
              :height: 15px

.. |incr| image:: functions_files/increasing.svg
              :width: 15px
              :height: 15px

.. |decr| image:: functions_files/decreasing.svg
              :width: 15px
              :height: 15px