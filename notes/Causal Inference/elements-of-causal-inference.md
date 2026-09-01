---
title: 'Notes: Elements of Causal Inference (Peters, Janzing, Schölkopf)'
from: markdown+tex_math_single_backslash
---


## Page 35

This seems a conceptually novel way of looking at equivalence of partial
pooling in causal models. Might be exact equivalence though.

Though it is worth noting that the independence seems to be a bigger
term than in statistics. Independence of cause and mechanism - mechanism
being $$p(t | a)$$ and cause being $$p(a).$$ If we start changing the
cause distribution then the t -\> a causal model does not keep the
mechanism intact.

The generative process or mechanism generating the effect from the cause
is indpeendent of the generative process or mechanism that is generating
the cause.

Three differnet visions of independence.

1.  Mechanistic - the light, the position of a chair and angle you are
    watching the chair are indpendent and there exists the assumption of
    generic viewpoint. You can change the light, position of the chair
    or perspective without being surprised.

2.  Information theoretic independence - learning about the structure of
    the conditional does not in any way inform you about the structure
    of the prior.

3.  Independence of noises - where is noise generated from with regards
    to the generative process. Not sure how this generalises last - it
    was mentioned that this can model or homogeneous processes as well.
    To break independence we can look at mechanisms or subprocesses
    $f_k$ such that the process is selected randomly when rv $X_t = k$.
    When we for example know that $X_{t+1} = X_t + 1$ then processes
    $f_k$ at $\tau = t$ or $\tau = t+1$ are not independent.

## Page 56

**Problem 3.5.**

``` python
from plots import hist_demo

hist_demo()
```

![](elements-of-causal-inference_files/figure-commonmark/cell-2-output-1.png)

[Problem 3.6](p36)

## Page 57

**Problem 3.8** Test ABC4

## Page 86

[Problem 4.15 (ANMs)](p415)

[Problem 4.16 (Maximum likelihood)](p416)

## Page 96

[Problem 5.1 (Independence of mechanisms)](p51)

[Helpers: convolution and deconvolution](deconv_helpers)
