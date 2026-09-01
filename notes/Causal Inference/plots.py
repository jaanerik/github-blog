"""Plots for the causal inference notes. Imported by causal_inference.notes.qmd."""

import numpy as np
import matplotlib.pyplot as plt


def hist_demo(n=200):
    """Problem 3.5: histogram of X = Y^2 + noise, Y ~ N(0,1)."""
    y = np.random.randn(n)
    x = y**2 + np.random.randn(n)
    plt.hist(x)
