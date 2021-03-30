# Implementing the Bayesian Online Changepoint Detection algorithm in MATLAB (CH5115 Course Project)

This is an implementation of the Bayesian Online Changepoint Detection algorithm (described in [this paper](https://arxiv.org/abs/0710.3742v1)) in MATLAB. It is heavily inspired from Gregory Gundersen's Python implementation of the same (blog post linked [here](http://gregorygundersen.com/blog/2019/08/13/bocd/), and the GitHub repo is linked [here](https://github.com/gwgundersen/bocd)).

In this implementation, since the mean and variance of the data are unknown, a normal gamma prior has been used.

All the resources used to make this implementation possible are saved in `sources.txt`. `NMRlogWell.mat` contains the data that was used to verify whether the algorithm works as intended.