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