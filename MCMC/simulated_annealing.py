
# Simulated annealing optimisation
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


def Simulated_Annealing(p, n, x0, y0, step):
    '''Samples p, the target distribution, n times according to 
    the Metropolis-Hastings algorithm while Q(x'|x) becomes a narrower 
    distribution. This means exploration is increasingly localised 
    along the chain '''

    global step_size
    step_size = step
    
    def anneal(step):
        global step_size 
        step_size *= 0.99
        return step_size
        
    x = x0
    y = y0
    
    points = np.zeros((n, 2)) # list of samples to fill

    for i in range(n):
        
        # propose new parameters from current state by sampling Q(x',y' | x,y)
        x_star, y_star = np.array([x, y]) + np.random.normal(scale=step_size, size=2)

        # Metropolis-Hastings condition
        if np.random.rand() < p(x_star, y_star) / p(x, y):
            x, y = x_star, y_star

        # update samples list
        points[i] = np.array([x, y])
        
        # reduce step size
        anneal(step_size)

    return points

def ptarget(x, y):
	'''the distribution we wish to sample'''
    return NotImplementedError