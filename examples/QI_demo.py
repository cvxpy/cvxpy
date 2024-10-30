import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    # Grabbing necessary imports
    import cvxpy as cp
    import numpy as np

    from cvxpy.atoms.affine.partial_trace import partial_trace
    return cp, np, partial_trace


@app.cell
def __(mo):
    mo.md(r"""## Nearest Correlation Matrix (in the sense of the QREP )""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        The first simple problem that we solve, is that of computing the nearest correlation matrix to a given symmetric matrix in the sense of the Quantum Relative Entropy (QREP) i.e., we want to find the closest SDP matrix with unit diagonal to our initial given matrix where distance is measured in the sense of the QREP --- this problem arises regularly in finance, where the correlations are between stocks and is usually solved in the sense of the Frobenius norm.

        \begin{alignat*}{3}
            &\text{minimize}& \left| \left| M - X \right| \right|_{\texttt{QRE}}\\
            &\text{subject to }& \mathop{\text{ diag}}(X)=e\\
            &                 & X\succeq 0\\
        \end{alignat*}

        where $M$ is a known symmetric matrix

        This problem may be translated to CVXPY almost literally as can be seen below
        """
    )
    return


@app.cell
def __(cp, n, np):
    n_corr = 4 # Choose `n` as per your taste
    np.random.seed(0)
    M_corr = np.random.randn(n_corr, n_corr)
    M_corr = M_corr @ M_corr.T
    X_corr = cp.Variable(shape=(n_corr, n_corr), symmetric=True)
    obj_corr = cp.Minimize(cp.quantum_rel_entr(M_corr, X_corr))
    cons_corr = [
                cp.diag(X_corr) == np.ones((n,)),
                X_corr >> 0
                ]
    prob_corr = cp.Problem(obj_corr, cons_corr)
    prob_corr.solve(solver='MOSEK')
    return M_corr, X_corr, cons_corr, n_corr, obj_corr, prob_corr


@app.cell
def __(mo):
    mo.md(r"""## Capacity of a **«classical-quantum»** channel""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        The next problem we will solve is that computing the capacity of a so-called, `«classical-quantum»` channel.

        For a finite input alphabet $\chi$ and a finite-dimensional Hilbert space $A$, a classical-quantum channel is a mapping $\Phi: \chi\to D(A)$ which maps symbols $x\in\chi$ to density operators $\Phi(x)$. The capacity of such a channel. The capacity of such a channel can be computed as the solution of an optimization problem:

        \begin{align*}
        &\mathop{\text{maximize}}_{p\in\mathbb{R}^{\chi}}\quad H \left( \sum_{x\in\chi}p(x)\Phi(x) \right) - \sum_{x\in\chi}p(x)H \left( \Phi(x) \right)\\
        &\mathop{\text{subject to}}\quad  p\geq 0, \sum_{x\in\chi}p(x) = 1
        \end{align*}

        Where $H(.)$ is the Von Neumann entropy.

        The above problem can be implemented fairly straightforwardly in CVXPY using a mixture of standard CVXPY functionality and the $\texttt{von\_neumann\_entr}$ atom. It also requires the $\texttt{randRho}$ routine from the `qubit` MATLAB package which generates random density matrices (which in turn relies on the $\texttt{randH}$ routine which generates random hermitian matrices)

        """
    )
    return


@app.cell
def __(np):
    def randH(n: int):
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        return (A + A.conj().T)/2

    def randRho(n: int):
        p = 10 * randH(n)
        p = (p @ p.conj().T)/np.trace(p @ p.conj().T)
        return p
    return randH, randRho


@app.cell
def __(cp, np, randRho):
    n_cq = 4

    rho1_cq = randRho(n_cq)
    rho2_cq = randRho(n_cq)
    H1_cq = cp.von_neumann_entr(rho1_cq)
    H2_cq = cp.von_neumann_entr(rho2_cq)

    p1_cq = cp.Variable()
    p2_cq = cp.Variable()

    obj_cq = cp.Maximize((cp.von_neumann_entr(p1_cq * rho1_cq + p2_cq * rho2_cq) - p1_cq * H1_cq - p2_cq * H2_cq)/np.log(2))
    cons_cq = [
        p1_cq >= 0,
        p2_cq >= 0,
        p1_cq + p2_cq == 1
    ]

    prob_cq = cp.Problem(obj_cq, cons_cq)
    prob_cq.solve(solver='MOSEK')
    return (
        H1_cq,
        H2_cq,
        cons_cq,
        n_cq,
        obj_cq,
        p1_cq,
        p2_cq,
        prob_cq,
        rho1_cq,
        rho2_cq,
    )


@app.cell
def __(mo):
    mo.md(r"""## Entanglement-assisted classical capacity of a quantum channel""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        The next problem that we solve is that of computing the so-called _«entanglement-assisted classical capacity»_ of a quantum channel, $\Phi$, which quantifies the amount of classical bits that can be transmitted reliably through it, if, the receiver and the transmitter are allowed to share an arbitrary entangled state.

        We require the notion of the $\textit{mutual-information}$ of the channel $\Phi$ for some input state $\rho$, defined as follows:

        Let $U: A\to B\otimes E$ be a Stinespring isometry for $\Phi$ with environment $E$ i.e. such that $\Phi(X) = \mathop{\text{tr}}_{E} \left[ UXU^{*} \right]$ for any operator $X$ on $A$. Then $I(\rho, \Phi)$ is defined as:

        \begin{equation*}
        I(\rho, \Phi) := H(B | E)_{U\rho U^{*}} + H(B)_{U\rho U^{*}}
        \end{equation*}

        Where $H(B|E)$ denotes the conditional entropy.

        We call upon one final property of the above defined mutual information, $I(\rho, \Phi)$ --- namely, that it is concave in $\rho$

        Coming back to the computation of the entanglement-assisted classical capacity. The problem can be shown to admit the following maximization expression:

        \begin{equation*}
        C_{\text{ea}} = \mathop{\text{max}}_{\rho\in D(A)} I(\rho, \Phi)
        \end{equation*}

        The above formula is the quantum analogue of the formula for the Shannon capacity of a classical channel.

        This can be implemented within CVXPY using a combination of traditional CVXPY functionalities and most notably, the $\texttt{quantum\_cond\_entr}$, $\texttt{von\_neumann\_entr}$ and $\texttt{partial\_trace}$ atoms.

        The quantum conditional entropy admits a definition in terms of the $\texttt{QRE}$ and the partial trace operator and has been implemented in CVXPY within `quantum_cond_entr`.

        Try it out below! (this problem will take longer to solve!)

        """
    )
    return


@app.cell
def __(cp, np):
    na_en, nb_en, ne_en = (2, 2, 2)
    AD_en = lambda gamma: np.array([[1, 0], [0, np.sqrt(gamma)], [0, np.sqrt(1-gamma)], [0, 0]])
    U_en = AD_en(0.2)

    rho_en = cp.Variable(shape=(na_en, na_en), hermitian=True)
    obj_en = cp.Maximize((cp.quantum_cond_entr(U_en @ rho_en @ U_en.conj().T, [nb_en, ne_en]) +
                       cp.von_neumann_entr(cp.partial_trace(U_en @ rho_en @ U_en.conj().T, [nb_en, ne_en], 1)))/np.log(2))
    cons_en = [
        rho_en >> 0,
        cp.trace(rho_en) == 1
    ]
    prob_en = cp.Problem(obj_en, cons_en)
    prob_en.solve(solver='MOSEK', verbose=True)
    return (
        AD_en,
        U_en,
        cons_en,
        na_en,
        nb_en,
        ne_en,
        obj_en,
        prob_en,
        rho_en,
    )


@app.cell
def __(mo):
    mo.md(r"""## Unassisted quantum capacity of a quantum channel""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        In this section, we solve the problem of estimating the unassisted quantum capacity of a quantum channel, we will require a bunch of definitions to setup the stage for defining the same --- first we define the notion of a _complimentary channel_.

        If $\Phi$ is a quantum channel from $A$ to $B$ with environment $E$ and an isometry representation $U:A\to B\otimes E$, then the complimentary channel, $\Phi^{c}: L(A)\to L(E)$ is given by:

        \begin{equation*}
        \Phi^{c}(\rho) = \mathop{\text{tr}}_{B} \left[ U\rho U^{*} \right]
        \end{equation*}

        Next, we define the notion of _Coherent information_:

        The coherent information of a channel $\Phi$ for the input $\rho$ is defined as:

        \begin{equation*}
        I_{c}(\rho, \Phi) := H \left( \Phi(\rho) \right) - H \left( \Phi^{c}(\rho) \right)
        \end{equation*}


        The unassisted quantum capacity $Q(\Phi)$ of quantum channels $\Phi$ is the number of qubits that can be reliably transmitted over $\Phi$ --- it can be computed via the following expression:

        \begin{equation*}
        Q(\Phi) = \lim_{n\to\infty}\mathop{\text{max}}_{\rho^{(n)}}\frac{1}{n}I_{c}(\rho^{(n)}, \Phi^{\otimes n})
        \end{equation*}

        Writing this problem out in CVXPY required the implementation of the $\texttt{applychan}$ method from the `qubit` MATLAB package. Additionally, this required the $\texttt{quantum\_cond\_entr}$ (which depends on the $\texttt{QRE}$ as discussed above).

        """
    )
    return


@app.cell
def __(cp, kron, np, partial_trace):
    def applychan(chan: np.array, rho: cp.Variable, rep: str, dim: tuple[int, int]):
        tol = 1e-10

        dimA, dimB, dimE = None, None, None
        match rep:
            case 'choi2':
                dimA, dimB = dim
            case 'isom':
                dimA = chan.shape[1]
                dimB = dim[1]
                dimE = int(chan.shape[0]/dimB)
                pass

        match rep:
            case 'choi2':
                arg = chan @ kron(rho.T, np.eye(dimB))
                rho_out = partial_trace(arg, [dimA, dimB], 0)
                return rho_out
            case 'isom':
                rho_out = partial_trace(chan @ rho @ chan.conj().T, [dimB, dimE], 1)
                return rho_out
    return (applychan,)


@app.cell
def __(applychan, cp, np):
    na_cc, nb_cc, ne_cc, nf_cc = (2, 2, 2, 2)
    AD_cc = lambda gamma: np.array([[1, 0],[0, np.sqrt(gamma)],[0, np.sqrt(1-gamma)],[0, 0]])
    gamma = 0.2
    U_cc = AD_cc(gamma)

    W_cc = AD_cc((1-2*gamma)/(1-gamma))

    Ic_cc = lambda rho: cp.quantum_cond_entr(
        W_cc @ applychan(U_cc, rho, 'isom', (na_cc, nb_cc)) @ W_cc.conj().T,
        [ne_cc, nf_cc], 1
    )/np.log(2)

    rho_cc = cp.Variable(shape=(na_cc, na_cc), hermitian=True)
    obj_cc = cp.Maximize(Ic_cc(rho_cc))
    cons_cc = [
        rho_cc >> 0,
        cp.trace(rho_cc) == 1
    ]
    prob_cc = cp.Problem(obj_cc, cons_cc)
    prob_cc.solve(solver='MOSEK', verbose=True)
    return (
        AD_cc,
        Ic_cc,
        U_cc,
        W_cc,
        cons_cc,
        gamma,
        na_cc,
        nb_cc,
        ne_cc,
        nf_cc,
        obj_cc,
        prob_cc,
        rho_cc,
    )


@app.cell
def __(mo):
    mo.md("""## Relative entropy of Entanglement""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        The final problem that we consider is that of solving for the _Relative entropy of Entanglement_.


        The $\textit{relative entropy of entanglement}$ is defined as the distance from a bipartite state on $D(A\otimes B)$, $\rho$, to the set of all separable states on $A\otimes B$ ($\texttt{Sep}$):

        \begin{equation*}
        \texttt{REE}(\rho) = \mathop{\text{min}}_{\tau\in\texttt{Sep}}D(\rho || \tau)
        \end{equation*}

        The set of all separable states is infamously hard to characterize, a popular relaxation for it is to replace $\rho\in\texttt{Sep}$ with imposing $\tau$ to have a positive partial transpose (the $\textit{PPT relaxation}$ of this problem as it's called)

        \begin{equation*}
        \texttt{REE}^{(1)}(\rho) = \mathop{\text{min}}_{\tau\in \text{PPT}}D(\rho || \tau)
        \end{equation*}

        """
    )
    return


@app.cell
def __(cp, np, randRho):
    na_ree, nb_ree = (2, 2)
    rho_ree = randRho(na_ree * nb_ree)
    tau_ree = cp.Variable(shape=(na_ree * nb_ree, na_ree * nb_ree), hermitian=True)
    obj_ree = cp.Minimize(cp.quantum_rel_entr(rho_ree, tau_ree, (3,3))/np.log(2))
    cons_ree = [tau_ree >> 0, cp.trace(tau_ree) == 1, cp.partial_transpose(tau_ree, [na_ree, nb_ree], 1) >> 0]
    prob_ree = cp.Problem(obj_ree, cons_ree)

    prob_ree.solve(solver='MOSEK', verbose=True)
    return cons_ree, na_ree, nb_ree, obj_ree, prob_ree, rho_ree, tau_ree


if __name__ == "__main__":
    app.run()
