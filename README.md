# Adaptive Deep Fourier Residual Method via Overlapping Domain Decomposition

This repository contains the source code implementation associated with the paper titled **"Adaptive Deep Fourier Residual Method via Overlapping Domain Decomposition."** The method detailed in the paper introduces an adaptive approach that utilizes deep Fourier residual networks in conjunction with overlapping domain decomposition for solving PDEs in 1D. 

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Examples](#examples)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## Abstract

The Deep Fourier Residual (DFR) method is a specific case of Variational Physics-Informed Neural Networks (VPINN) among a wide range of strategies for solving PDEs using Neural Networks (NNs). In the DFR method the loss function is an approximation to the dual norm of the weak residual of the PDE. This loss function ensures that reducing the loss during the training of an NN is equivalent to reducing the error in the solution at the same rate. In previous works, the calculation of the dual norm is based on a spectral representation of the dual norms of the test function space on rectangles. Here, we propose an extension of the DFR method to use adaptive strategies on general polygonal domains. We decompose the PDE domain Ω into rectangular subdomains, and the loss function is computed as the sum of local loss functions. We use the Dofler marking algorithm to adaptively refine the initial subdomain decomposition of Ω and increase the accuracy of the approximated solution on relevant regions of the domain.

## References 

**Deep Fourier Residual method for solving time-harmonic Maxwell's equations**
https://arxiv.org/abs/2305.09578

**A Deep Fourier Residual Method for solving PDEs using Neural Networks**
https://arxiv.org/abs/2210.14129



## Requirements

- Python 3.10.10
- TensorFlow 2.10
- NumPy

## Examples

We consider the following ODE in variational form: find $u\in H^1_0(0,\pi)$ satisfying the weak formulation of Poisson's equation, i.e., 

$\int_0^\pi u'(x)v'(x)-f(x)v(x)\,dx = 0 \qquad \forall v\in H^1_0(0,\pi),$

where $f$ is such that the exact solution is
$u^*(x)=x(x-\pi)\exp\left(-120\left(x-\frac{\pi}{2}\right)^2\right).$

The function $u^*$ is smooth but is mostly constant near the boundary and exhibits a prominent peak and a corresponding large derivative near $x=\frac{\pi}{2}$. 

## Authors 

Prof. Dr. Jamie M. Taylor. CUNEF Universidad, Madrid, Spain. (jamie.taylor@cunef.edu) 

Prof. Dr. Manuela Bastidas. University of the Basque Country (UPV/EHU), Leioa, Spain. / Universidad Nacional de Colombia, Medellín, Colombia. (manumnlb@gmail.com)

## Acknowledgments

Authors have received funding from the Spanish Ministry of Science and Innovation projects with references TED2021-132783B-I00, PID2019-108111RB-I00 (FEDER/AEI), and PDC2021-121093-I00 (MCIN / AEI / 10.13039 / 501100011033 / Next Generation EU), the ``BCAM Severo Ochoa'' accreditation of excellence CEX2021-001142-S / MICIN / AEI / 10.13039 / 501100011033; the Spanish Ministry of Economic and Digital Transformation with Misiones Project IA4TES (MIA.2021.M04.008 / NextGenerationEU PRTR); and the Basque Government through the BERC 2022-2025 program, the Elkartek project BEREZ-IA (KK-2023 / 00012),, and the Consolidated Research Group MATHMODE (IT1456-22) given by the Department of Education. 
