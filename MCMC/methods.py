

# visualisation and data manipulation methods for the MCMC sampling 
# algorithms.
#
# Copyright (C) 2017 Peter Mann
# 
# This file is part of `MCMC`, for sampling distributions using
# a variety of Markov chain Monte Carlo methods. 
#
# `MCMC` is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# `MCMC` is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with `MCMC`. If not, see <http://www.gnu.org/licenses/gpl.html>.


from matplotlib import pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import numpy as np

def plot_dist(x,y):
    ''' plotting environment we frequently use to visualise the 
    sampling algorithms.'''
    
    # style for seaborn
    sns.set(style="ticks", color_codes=True)
    plt.xkcd()

    # plot contour
    g = (sns.JointGrid(x, y, size=10)
     .plot_joint(sns.kdeplot, n_levels=6, cmap="BuPu", shade=True, shade_lowest=True))

    # plot marginals 
    g = g.plot_marginals(sns.kdeplot, color="b", shade=True, shade_lowest=True)

    # add scatter plot of data
    g = g.plot_joint(plt.scatter, c="b", s=30, linewidth=1)
    plt.plot(x, y, linestyle='-', marker='o', alpha=0.4, )
    
    # set axis labels
    g.set_axis_labels("$X$", "$Y$")
    
def traceplot(x):
    '''traceplot in the x-dimension with the cumulative mean (r)
    and the actual mean (--)'''
    sns.tsplot(x)
    l = list(accumulate(x))
    nl = [v/i for i, v in enumerate(l,1)]
    plt.plot(nl, 'r')
    plt.axhline(y=0.0, color='k', linestyle='--')
    plt.ylabel('x')
    plt.xlabel('Iteration')
    
def autocorrelation_plot(x, l):
    '''plots the autocorrelation of x to lag = l'''
    plt.acorr(x - np.mean(x), maxlags=l,  normed=True, usevlines=False);
    plt.xlim((0, 100))
    plt.ylabel('Autocorrelation')
    plt.xlabel('Lag')