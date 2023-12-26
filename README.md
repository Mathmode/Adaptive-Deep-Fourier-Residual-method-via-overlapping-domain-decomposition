# Adaptive Deep Fourier Residual Method via Overlapping Domain Decomposition

This repository contains the source code implementation associated with the paper titled **"Adaptive Deep Fourier Residual Method via Overlapping Domain Decomposition."** The method detailed in the paper introduces an adaptive approach that utilizes deep Fourier residual networks in conjunction with overlapping domain decomposition for solving PDEs in 1D. 

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## Abstract


## Requirements

- Python 3.10.10
- TensorFlow 2.10
- NumPy

## Examples

We consider the following ODE in variational form: find $u\in H^1_0(0,\pi)$ satisfying Eq.~\eqref{eq:boundsing}, where $f$ is such that the exact solution is
$$ u^*(x)=x(x-\pi)\exp\left(-120\left(x-\frac{\pi}{2}\right)^2\right).$$

The function $u^*$ is smooth but is mostly constant near the boundary and exhibits a prominent peak and a corresponding large derivative near $x=\frac{\pi}{2}$. 
