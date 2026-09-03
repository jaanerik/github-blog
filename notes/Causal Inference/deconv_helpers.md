**Helpers: convolution and deconvolution.**

**Helper 1 (convolution as a matrix, finite case).** Let $X\in\{1,\dots,m\}$ with probability vector $p$, and noise $N\in\{0,1\}$ with $P(N=0)=P(N=1)=\tfrac12$, independent of $X$. Then $Y=X+N\in\{1,\dots,m+1\}$ has

$$
r(y) = \tfrac12\,p(y) + \tfrac12\,p(y-1), \qquad p(0):=p(m+1):=0,
$$

i.e. $r = A p$ for the $(m+1)\times m$ banded matrix $A$ with $\tfrac12$ on the diagonal and subdiagonal. Deconvolution is solving $Ap=r$ for $p$: $p(1)=2r(1)$, $p(2)=2r(2)-p(1)$, …, a back-substitution that exists and is unique. Note (i) $A$ depends only on the noise law; (ii) the recipe does not care what $p$ is; (iii) errors in $r$ double at every step (instability).

**Helper 2 (Gaussians convolve by adding variances).** If $X\sim\mathcal N(\mu,a^2)$, $N\sim\mathcal N(0,b^2)$, independent, then $Y=X+N\sim\mathcal N(\mu,a^2+b^2)$. (Prove once: either $\int\varphi_a(x-\mu)\,\varphi_b(y-x)\,dx$ by completing the square, or via characteristic functions $e^{i\mu t-a^2t^2/2}\cdot e^{-b^2t^2/2}$.) Hence within the Gaussian family *deconvolution is “subtract the noise variance”*: given $Y\sim\mathcal N(\mu,c^2)$ and known $b$, the input was $\mathcal N(\mu,c^2-b^2)$. It exists iff $c^2\ge b^2$: a $P_Y$ narrower than the noise cannot have been produced by any $P_X$. Again the map $c^2\mapsto c^2-b^2$ is fixed by $b$ alone.

**Helper 3 (Fourier makes it linear algebra).** For densities, $\widehat{p*q}=\widehat p\,\widehat q$ with $\widehat p(t):=\int e^{itx}p(x)\,dx$; this is the independence identity $\mathbb E e^{it(X+N)}=\mathbb E e^{itX}\,\mathbb E e^{itN}$ [T6 in `cond_exp.pdf`]. So convolution is a *diagonal* operator in the Fourier basis, with “eigenvalue” $\widehat q(t)$ at frequency $t$, and deconvolution divides by it: $\widehat p=\widehat r/\widehat q$. Helper 2 is this with $\widehat q(t)=e^{-b^2t^2/2}$: dividing subtracts $b^2$ from the variance. Ill-posedness $=$ $\widehat q(t)\to0$ as $|t|\to\infty$, so high-frequency error in $r$ is amplified without bound.

**Common thread.** Deconvolution is a map on distributions, $P_Y\mapsto P_X$, determined by the noise law only, and it is linear. Keep this in mind for Problem 5.1 b).

**Exercise (stays inside Helper 2).** For $X\sim\mathcal N(0,a^2)$, $N\sim\mathcal N(0,b^2)$, compute the conditional law $P_{X\mid Y=y}$ (Gaussian; same computation as Problem 3.6 in `p36.tex`). Then ask what map on distributions $P_{X\mid Y}$ induces via $P_X(B)=\int P_{X\mid Y=y}(B)\,P_Y(dy)$, and compare its ingredients with the deconvolution map’s ingredients.

**Solution** Let us fix $y$ and write

$$
p(x|y) = \frac{p(x)p(y|x)}{p(y)} = \frac{1/a \phi(x/a) \cdot 1/b \phi((y-x)/b)}{1/\sqrt{a^2+b^2} \phi(y/\sqrt{a^2+b^2})} = c(y) \exp \left[ -(x/a)^2/2 - (y-x)^2/2b^2 + y^2/2(a+b) \right]
$$

Now note that $y^2/(a+b) \text{ and }, y^2/2b^2$ can be subsumed into some new $\Tilde{c}(y)$. Now focus on the exponent terms containing $x$

$$
\begin{align*}
  p(x|y) &= c(y) \exp\left[ -x^2/2a^2 + xy/b^2 - x^2/2b^2 \right] \\
  &= \exp \left[ \frac{-b^2 x^2 + 2a^2 xy - a^2 x^2}{2a^2b^2} \right] \\
  &= \exp \left[ \frac{- (\sqrt{a^2+b^2}x + \ldots )}{2a^2b^2} \right] \\
\end{align*}
$$

**Results side by side** ($X\sim\mathcal N(0,a^2)$, $N\sim\mathcal N(0,b^2)$, $Y=X+N$, $c^2:=a^2+b^2$):

$$
\begin{align*}
  P_{X\mid Y=y} &= \mathcal N\!\Big(\frac{a^2}{a^2+b^2}\,y,\ \frac{a^2b^2}{a^2+b^2}\Big)
  && \text{(conditional; mean depends on } y\text{, variance } <a^2, <b^2) \\
  P_X &= \mathcal N\big(0,\ c^2-b^2\big)
  && \text{(deconvolution of } P_Y=\mathcal N(0,c^2)\text{; ingredients: } P_Y \text{ and } b \text{ only)}
\end{align*}
$$

Both recover $P_X$ from $P_Y$: the second directly, the first via $P_X(B)=\int P_{X\mid Y=y}(B)\,P_Y(dy)$. They are different maps on distributions: the first has $a$ (i.e. $P_X$) baked into its coefficients, the second does not.

**The two maps.** Fix the model $X\sim\mathcal N(\mu,a^2)$, $N\sim\mathcal N(0,b^2)$, $N\perp\!\!\!\perp X$, $Y=X+N$, and write $\kappa := a^2/(a^2+b^2)$, $v := a^2b^2/(a^2+b^2)$, $m(y) := \mu + \kappa(y-\mu)$, so that $P_{X\mid Y=y} = \mathcal N(m(y), v)$. Both maps take a probability measure $\tilde Q$ on the $y$-axis to one on the $x$-axis:

$$
\begin{align*}
  C(\tilde Q) &:= \mathcal N(\tilde\mu,\ \tilde c^2 - b^2)
  \quad\text{for } \tilde Q = \mathcal N(\tilde\mu,\tilde c^2)
  && \text{(deconvolution; built from } b \text{ only)} \\
  K(\tilde Q)(B) &:= \int \mathcal N(m(y), v)(B)\,\tilde Q(dy)
  && \text{(kernel map; built from } \mu,\kappa,v\text{, i.e.\ from } P_X\text{)}
\end{align*}
$$

Here $\mu, a, b$ (hence $\kappa, v, m$) are frozen into the maps; $\tilde\mu, \tilde c^2$ parameterize the varying input.

With the help of Claim 1 (`anm_claims.pdf`) and defining $V := m(\tilde{Y}) + Z$ with $\tilde Y\sim\tilde Q$, $Z\sim\mathcal N(0,v)$, $Z\perp\!\!\!\perp\tilde Y$, we can show that $P_V = K(\tilde{Q})$, with the distribution

$$
V \sim \mathcal{N}(\mu(1-\kappa) + \kappa \tilde{\mu},\ \kappa^2 \tilde{c}^2 + v).
$$

Since $C(\tilde Q) = \mathcal N(\tilde\mu,\ \tilde c^2-b^2)$ while $K(\tilde Q) = P_V = \mathcal{N}(\mu(1-\kappa) + \kappa \tilde{\mu},\ \kappa^2 \tilde{c}^2 + v)$, the two maps differ whenever $\tilde{\mu} \ne \mu$; moreover for $\tilde c^2 < b^2$ the map $C$ is undefined while $K$ is not, and for $\tilde\mu = \mu$ the variances agree only at $\tilde c^2 = a^2+b^2$. The maps agree exactly at $\tilde Q = P_Y$, where both return $P_X$.
