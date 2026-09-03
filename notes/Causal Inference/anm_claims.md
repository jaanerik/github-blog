---
title: "Problem 4.15 — the two claims behind \"no ANM from $Y$ to $X$\""
---

Setting (Definition 4.4, p. 67): an ANM from $Y$ to $X$ means there exist a measurable $g:\mathbb{R}\to\mathbb{R}$ and a random variable $M_X$ with
$$
X = g(Y) + M_X, \qquad M_X \perp\!\!\!\perp Y .
$$
Everything below assumes such $(g, M_X)$ exists and derives a contradiction.

## What "conditional distribution given $Y=y$" means

For continuous $Y$ the event $\{Y=y\}$ has probability $0$, so $P(X\in A \mid Y=y)$ cannot be defined by division. Instead, a *regular conditional distribution* of $X$ given $Y$ is a family of probability measures $\{\kappa_y\}_{y\in\mathbb{R}}$ on $\mathbb{R}$ (measurable in $y$) such that
$$
P(X\in A,\ Y\in B) \;=\; \int_B \kappa_y(A)\, P_Y(dy)
\qquad \text{for all Borel } A, B. \tag{$\ast$}
$$
Such a family exists, and it is unique up to $P_Y$-null sets of $y$: if $\{\kappa_y\}$ and $\{\kappa'_y\}$ both satisfy $(\ast)$ then $\kappa_y=\kappa'_y$ for $P_Y$-a.e. $y$. We write $P_{X\mid Y=y} := \kappa_y$.

So the sentence "$P_{X\mid Y=y}$ is the law of $g(y)+M_X$" means: the family $\kappa_y(A) := P\big(g(y)+M_X \in A\big)$ satisfies $(\ast)$.

## Claim 1 (independence $\Rightarrow$ conditionals are shifts of one law)

\begin{claim}
If $X = g(Y)+M_X$ with $M_X \perp\!\!\!\perp Y$, then $\kappa_y(A) := P\big(g(y)+M_X\in A\big)$ is a regular conditional distribution of $X$ given $Y$. Equivalently: for $P_Y$-a.e.\ $y$, $\;P_{X\mid Y=y} = P_{M_X}(\,\cdot - g(y))$, the law of $M_X$ translated by $g(y)$.
\end{claim}

*Proof.* Fix Borel $A,B$. Because $M_X\perp\!\!\!\perp Y$, the joint law of $(Y, M_X)$ is the product $P_Y\otimes P_{M_X}$. Then
$$
P(X\in A,\ Y\in B)
= P\big(g(Y)+M_X\in A,\ Y\in B\big)
= \int\!\!\int \mathbf 1_A\big(g(y)+m\big)\,\mathbf 1_B(y)\; P_{M_X}(dm)\,P_Y(dy).
$$
The inner integral is $P\big(g(y)+M_X\in A\big) = \kappa_y(A)$, so the right side is $\int_B \kappa_y(A)\,P_Y(dy)$, i.e. $(\ast)$ holds. $\square$

(This is the only place Fubini/Tonelli is used: writing the probability under a product measure as an iterated integral. Independence is what makes the joint law a product.)

## Claim 2 (a shift-invariant statistic must be constant)

Let $T$ be any map from probability measures on $\mathbb{R}$ to $[0,\infty]$ that is **translation invariant**: $T\big(\mu(\cdot - c)\big) = T(\mu)$ for all $c\in\mathbb{R}$. Examples: $\operatorname{Var}(\mu)$; the Lebesgue measure of the support of $\mu$; the interquartile range.

\begin{claim}
Under the hypothesis of Claim 1, $y \mapsto T(P_{X\mid Y=y})$ is $P_Y$-a.e.\ equal to the constant $T(P_{M_X})$.
\end{claim}

*Proof.* By Claim 1, for $P_Y$-a.e. $y$, $P_{X\mid Y=y} = P_{M_X}(\cdot - g(y))$, hence $T(P_{X\mid Y=y}) = T(P_{M_X})$ by translation invariance. $\square$

## Contradiction for Problem 4.15(a)

From the joint density (uniform, value $\tfrac12$, on the parallelogram $\{1\le x\le 3,\ |y-2x|\le\tfrac12\}$),
$$
P_{X\mid Y=y} = \mathcal U\!\left[\max\!\big(1,\tfrac y2-\tfrac14\big),\ \min\!\big(3,\tfrac y2+\tfrac14\big)\right]
\qquad\text{for } y \in [\tfrac32,\tfrac{13}{2}] .
$$
(This is a genuine version of the conditional distribution: one checks $(\ast)$ by integrating the joint density; it is unique a.e. by the uniqueness statement above.)

Take $T = $ support length (or $T=\operatorname{Var}$, which is $\text{length}^2/12$ for a uniform law):

- $y\in(\tfrac52,\tfrac{11}2)$: length $\tfrac12$;
- $y\in(\tfrac32,\tfrac52)$: length $\tfrac y2-\tfrac34 < \tfrac12$.

Both intervals have positive $P_Y$-mass ($p_Y(y) = \tfrac12\cdot\text{length} > 0$ there). So $y\mapsto T(P_{X\mid Y=y})$ is **not** a.e. constant, contradicting Claim 2. Hence no $(g,M_X)$ with $M_X\perp\!\!\!\perp Y$ exists: $P_{X,Y}$ admits no ANM from $Y$ to $X$.

## Remarks

- Fubini is not "needed for the idea"; it is needed only to make Claim 1 a theorem rather than a picture, because conditioning on a null event is defined through $(\ast)$.
- Variance is not needed either; any translation-invariant $T$ works. Support length is closest to the "draw the support" hint in the exercise.
- Theorem 4.5 (pp. 67–68) is not applicable: it needs strictly positive, three-times differentiable densities; the uniform noise here has neither.
