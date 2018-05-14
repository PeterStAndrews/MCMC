

# Gibbs sampler
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

import numpy as np 

def Gibbs_Sampler(mean, var, n, x0, y0):
    '''Gibbs sampling algorithm for bivariate normal distribution. 
    The conditional distributions for the parameters are sequentially
    sampled and '''
    
    # first guess
    x = x0
    y = y0
    
    x_samples = [x]
    y_samples = [y]

    for i in range(n):
        
        # sample x given y
        x = Q_x_mid_y(y, mean, var)
        
        # sample y given x
        y = Q_y_mid_x(x, mean, var)
        
        # append samples with update
        x_samples.append(x)
        y_samples.append(y)

    return x_samples, y_samples

def  Q_x_mid_y(y, mean, var):
    '''The conditional of x given y.'''
    return NotImplementedError


def Q_y_mid_x(x, mean, var):
    '''The conditional of y given x.'''
    return NotImplementedError