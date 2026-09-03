---
title: "Conditional expectation and regular conditional distributions, built from TNT II"
---

References in brackets are to Lember, *Tõenäosusteooria II* (tnt2-24). Throughout, $(\Omega,\mathcal{F},P)$ is a probability space, $X,Z:\Omega\to\mathbb{R}$ random variables, and $\mathcal{B}=\mathcal{B}(\mathbb{R})$ the Borel $\sigma$-algebra.

# Tools used (stated in full)

**T1 — Measurability [§4.1].** For measurable spaces $(S,\Sigma)$, $(S',\Sigma')$, a map $T:S\to S'$ is $\Sigma/\Sigma'$-measurable if $T^{-1}(B)\in\Sigma$ for every $B\in\Sigma'$. Compositions of measurable maps are measurable: if $T$ is $\Sigma/\Sigma'$-measurable and $f$ is $\Sigma'/\Sigma''$-measurable, then $f\circ T$ is $\Sigma/\Sigma''$-measurable, since $(f\circ T)^{-1}(B)=T^{-1}(f^{-1}B)$. A random variable is an $\mathcal{F}/\mathcal{B}$-measurable $X:\Omega\to\mathbb{R}$, and $\sigma(X):=\{X^{-1}(B):B\in\mathcal{B}\}$ is the smallest $\sigma$-algebra making $X$ measurable.

**T2 — Pushforward (jaotus) [§7.11].** If $T:S\to S'$ is $\Sigma/\Sigma'$-measurable and $\mu$ a measure on $(S,\Sigma)$, then $\mu T^{-1}(A'):=\mu(T^{-1}A')$ defines a measure on $(S',\Sigma')$. For a random variable $X$, the law of $X$ is $P_X:=PX^{-1}$ on $(\mathbb{R},\mathcal{B})$.

**T3 — Change of variables [Thm 7.23].** With $T,\mu$ as in T2 and $f\in m\Sigma'^{+}$ (or $f\circ T\in L^1(\mu)$):
$$\int_{T^{-1}A'} f(Ts)\,\mu(ds)=\int_{A'} f(s')\,\mu T^{-1}(ds')\qquad\forall A'\in\Sigma'.$$
In particular $\mathbb{E}[h(X)\mathbf 1_B(X)]=\int_B h(x)\,P_X(dx)$.

**T4 — Absolute continuity and Radon–Nikodym [Def 7.21, §7.10.5].** $\nu\ll\mu$ means: $\mu(A)=0\Rightarrow\nu(A)=0$. If $\mu,\nu$ are $\sigma$-finite measures on $(S,\Sigma)$ and $\nu\ll\mu$, there exists an $f\in m\Sigma^{+}$ (the density, $f=\frac{d\nu}{d\mu}$) with
$$\nu(A)=\int_A f\,d\mu\qquad\forall A\in\Sigma.$$

**T5 — Uniqueness of densities [Thm 7.16].** If $f,g\in m\Sigma^{+}$ satisfy $\int_A f\,d\mu=\int_A g\,d\mu$ for all $A\in\Sigma$ (and the common measure is $\sigma$-finite), then $f=g$ $\mu$-almost everywhere.

**T6 — Independence [§5].** Random variables $U,V$ are independent, $U\perp\!\!\!\perp V$, if $P(U\in A, V\in B)=P(U\in A)P(V\in B)$ for all $A,B\in\mathcal{B}$. Consequence: for integrable independent $U,V$, $\mathbb{E}[UV]=\mathbb{E}U\,\mathbb{E}V$ [§8]. More generally, a random variable $Z$ is independent of a $\sigma$-algebra $\mathcal{G}$ if $\sigma(Z)$ and $\mathcal{G}$ are independent, i.e. $P(\{Z\in A\}\cap G)=P(Z\in A)P(G)$ for all $A\in\mathcal{B}$, $G\in\mathcal{G}$.

**T7 — Dynkin's $\pi$--$\lambda$ theorem [§1.5].** If two probability measures agree on a $\pi$-system (a family closed under finite intersections) generating a $\sigma$-algebra, they agree on the whole $\sigma$-algebra. Used here only once, in the uniqueness discussion.

**T8 — Doob–Dynkin lemma** (not in tnt2; see e.g. Kallenberg, Lemma 1.14). Every $\sigma(X)/\mathcal{B}$-measurable $W:\Omega\to\mathbb{R}$ is of the form $W=h(X)$ for some $\mathcal{B}/\mathcal{B}$-measurable $h:\mathbb{R}\to\mathbb{R}$. (Idea: true for indicators $W=\mathbf 1_{X^{-1}(B)}=\mathbf 1_B(X)$ by definition of $\sigma(X)$; extend by the standard machine — simple functions, monotone limits.)


# Part A — Regular conditional distribution: what to verify

\begin{defn}[regular conditional distribution of $Z$ given $X$]
A map $\kappa:\mathbb{R}\times\mathcal{B}\to[0,1]$, written $(x,A)\mapsto\kappa_x(A)$, is a regular conditional distribution of $Z$ given $X$ if
\begin{enumerate}
\item[(K1)] for every $x\in\mathbb{R}$, $A\mapsto\kappa_x(A)$ is a probability measure on $(\mathbb{R},\mathcal{B})$;
\item[(K2)] for every $A\in\mathcal{B}$, $x\mapsto\kappa_x(A)$ is $\mathcal{B}/\mathcal{B}$-measurable;
\item[(K3)] for all $A,B\in\mathcal{B}$:
$$P(Z\in A,\ X\in B) \;=\; \int_B \kappa_x(A)\,P_X(dx).$$
\end{enumerate}
\end{defn}

Reading of (K3): the left side is a joint probability; the right side says "average, over $x\in B$ (weighted by the law of $X$), the probability that $Z\in A$ when $X=x$". (K3) is the only condition that ties $\kappa$ to $Z$; (K1)–(K2) say $\kappa$ is a genuine family of distributions ("regular") and that integrating it makes sense.

Given $\kappa$, define for $P_X$-a.e. $x$
$$\mathbb{E}[Z\mid X=x] := \int_{\mathbb{R}} z\,\kappa_x(dz).$$

**Uniqueness (why "a.e.").** Fix $A$. Both $x\mapsto\kappa_x(A)$ and $x\mapsto\kappa'_x(A)$ are densities, w.r.t. $P_X$, of the same finite measure $B\mapsto P(Z\in A, X\in B)$ on $(\mathbb{R},\mathcal{B})$; by T5 they agree $P_X$-a.e. (The null set depends on $A$; one then uses countably many $A$ — e.g. $A=(-\infty,q]$, $q\in\mathbb{Q}$ — and the $\pi$--$\lambda$ theorem [T7] to get a single null set outside of which $\kappa_x=\kappa'_x$ as measures. This is the subtle step; accept it on first reading.)

## The case $Z=f(X)$, $f$ Borel: verify $\kappa_x := \delta_{f(x)}$

Claim: $\kappa_x=\delta_{f(x)}$ (Dirac mass at $f(x)$) is a regular conditional distribution of $f(X)$ given $X$.

- (K1) $\delta_{f(x)}$ is a probability measure [§2 examples]. $\checkmark$
- (K2) $\kappa_x(A)=\delta_{f(x)}(A)=\mathbf 1_A(f(x))=\mathbf 1_{f^{-1}(A)}(x)$. Since $f$ is $\mathcal{B}/\mathcal{B}$-measurable, $f^{-1}(A)\in\mathcal{B}$, and the indicator of a Borel set is Borel-measurable. $\checkmark$
- (K3) Using only the definition of the pushforward $P_X(C)=P(X^{-1}C)$ [T2]:
$$P(f(X)\in A,\ X\in B) = P\big(X\in f^{-1}(A)\cap B\big) = P_X\big(f^{-1}(A)\cap B\big) = \int_B \mathbf 1_{f^{-1}(A)}(x)\,P_X(dx) = \int_B \delta_{f(x)}(A)\,P_X(dx).\\ \checkmark$$

Hence $\mathbb{E}[f(X)\mid X=x]=\int z\,\delta_{f(x)}(dz)=f(x)$. No Fubini is needed here; Fubini entered in `anm_claims.md` only because there the conditional law was $\delta_{g(y)}*P_{M_X}$, a convolution, and the joint law of $(Y,M_X)$ had to be written as a product.

## The case $Z=N_Y$ with $N_Y\perp\!\!\!\perp X$: verify $\kappa_x := P_{N_Y}$ (does not depend on $x$)

- (K1), (K2) trivial (constant in $x$).
- (K3) $P(N_Y\in A, X\in B)=P(N_Y\in A)\,P(X\in B)=P_{N_Y}(A)\int_B 1\,P_X(dx)=\int_B \kappa_x(A)\,P_X(dx)$, the first equality being the definition of independence [T6]. $\checkmark$

Hence $\mathbb{E}[N_Y\mid X=x]=\int z\,P_{N_Y}(dz)=\mathbb{E}[N_Y]=\mu_{N_Y}$. This is the second (why?) of Solution a).

# Part B — $\mathbb{E}[Z\mid\mathcal{G}]$: the definition via Radon–Nikodym

Let $\mathcal{G}\subseteq\mathcal{F}$ be a sub-$\sigma$-algebra ("the information") and $Z\in L^1(\Omega,\mathcal{F},P)$.

**Construction.** Define on the *smaller* measurable space $(\Omega,\mathcal{G})$ the signed measure
$$\nu(G) := \int_G Z\,dP,\qquad G\in\mathcal{G}$$
(for $Z\ge 0$ this is a finite measure [§7]; in general split $Z=Z^+-Z^-$). If $P(G)=0$ then $\nu(G)=0$, i.e. $\nu\ll P|_{\mathcal{G}}$ [T4]. Radon–Nikodym [T4], applied on $(\Omega,\mathcal{G},P|_{\mathcal{G}})$, gives a $\mathcal{G}$-measurable $W$ with $\nu(G)=\int_G W\,dP$ for all $G\in\mathcal{G}$, unique $P$-a.e. [T5].

\begin{defn}
$\mathbb{E}[Z\mid\mathcal{G}]:=W$. Equivalently, $\mathbb{E}[Z\mid\mathcal{G}]$ is the $P$-a.s. unique random variable with
\begin{enumerate}
\item[(C1)] $\mathbb{E}[Z\mid\mathcal{G}]$ is $\mathcal{G}$-measurable;
\item[(C2)] $\displaystyle\int_G \mathbb{E}[Z\mid\mathcal{G}]\,dP=\int_G Z\,dP$ for every $G\in\mathcal{G}$.
\end{enumerate}
\end{defn}

**The point of (C1).** $Z$ itself satisfies (C2) trivially. If (C1) were absent, the definition would be empty: $\mathbb{E}[Z\mid\mathcal{G}]=Z$ always. The content is that we demand a $\mathcal{G}$-measurable object — a "coarse-grained" version of $Z$ that can only vary as much as $\mathcal{G}$ can distinguish — while preserving all $\mathcal{G}$-averages. That is why Radon–Nikodym had to be applied on $(\Omega,\mathcal{G})$ and not on $(\Omega,\mathcal{F})$.

**Sanity check against [p. 187].** If $\mathcal{G}=\sigma(N)$ for a discrete $N$, the atoms of $\mathcal{G}$ are $\{N=n\}$; (C1) forces $\mathbb{E}[Z\mid\mathcal{G}]$ to be constant on each atom, and (C2) with $G=\{N=n\}$ identifies that constant as $\frac{1}{P(N=n)}\int_{\{N=n\}}Z\,dP=\mathbb{E}[Z\mid N=n]$. Taking $G=\Omega$ in (C2) recovers $\mathbb{E}Z=\sum_n\mathbb{E}[Z\mid N=n]P(N=n)$, the formula used in Lemma 13.1.

\begin{lemma}[taking out what is known]
If $Z$ is $\mathcal{G}$-measurable and integrable, then $\mathbb{E}[Z\mid\mathcal{G}]=Z$ a.s.
\end{lemma}
*Proof.* $Z$ satisfies (C1) by hypothesis and (C2) trivially. By uniqueness, $\mathbb{E}[Z\mid\mathcal{G}]=Z$ a.s. $\square$

\begin{lemma}[independence drops out]
If $Z\in L^1$ is independent of $\mathcal{G}$ (i.e.\ $\sigma(Z)$ and $\mathcal{G}$ are independent [T6]), then $\mathbb{E}[Z\mid\mathcal{G}]=\mathbb{E}Z$ a.s.
\end{lemma}
*Proof.* The constant $\mathbb{E}Z$ satisfies (C1). For (C2): $\int_G Z\,dP=\mathbb{E}[Z\mathbf 1_G]=\mathbb{E}Z\cdot P(G)=\int_G \mathbb{E}Z\,dP$, using $\mathbb{E}[UV]=\mathbb{E}U\,\mathbb{E}V$ for independent integrable $U,V$ [T6]. $\square$

## From $\mathbb{E}[Z\mid\sigma(X)]$ to $\mathbb{E}[Z\mid X=x]$

Take $\mathcal{G}=\sigma(X)=\{X^{-1}(B):B\in\mathcal{B}\}$ [§4]. The Doob–Dynkin lemma [T8] says: every $\sigma(X)$-measurable $W$ is of the form $W=h(X)$ for some Borel $h$. Define $\mathbb{E}[Z\mid X=x]:=h(x)$ ($P_X$-a.e. well defined).

This matches Part A: if $\kappa$ satisfies (K1)–(K3) and $h(x):=\int z\,\kappa_x(dz)$, then $h(X)$ is $\sigma(X)$-measurable and, for $G=\{X\in B\}$, (K3) plus change of variables [T3] give $\int_G h(X)\,dP=\int_B h(x)\,P_X(dx)=\int_{\{X\in B\}}Z\,dP$, i.e. (C2). So the two routes define the same object.

# Application to Solution a) of Problem 4.16

$$\mathbb{E}[Y\mid X]=\mathbb{E}[f(X)\mid X]+\mathbb{E}[N_Y\mid X]=f(X)+\mathbb{E}N_Y\quad\text{a.s.},$$
the first term by Lemma 1 (with $\mathcal{G}=\sigma(X)$; $f(X)$ is $\sigma(X)$-measurable because $(f\circ X)^{-1}(B)=X^{-1}(f^{-1}B)\in\sigma(X)$), the second by Lemma 2. Evaluating at $X=x$: $\mathbb{E}[Y\mid X=x]=f(x)+\mu_{N_Y}$. Part A gives the same conclusion via $\kappa_x=\delta_{f(x)}$ and $\kappa_x=P_{N_Y}$.
