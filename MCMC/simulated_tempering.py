

# Simulated tempering sampler
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
import random

def Simulated_Tempering(p, step_sizes, n, u, x0, y0):
    '''samples p using a single Markov chain of length n
    that is updated via the Metropolis-Hastings algorithm,
    while u times the step-size is dynamically udpated
    from a distribution of potential standard deviations 
    (step_sizes) according to the Metropolis condition.'''
    
    # choose initial step_size
    step_size = random.choice(step_sizes)
    
    # append first guess to record of states
    x_list = [x0]
    y_list = [y0]
    
    # step size record
    j = 0
    update_record = []
    
    for update in range(u):
        
        for i in range(n):
        
            # propose new parameters from current state by sampling Q(x',y' | x,y)
            x_star = x_list[-1] + np.random.normal(scale=step_size)
            y_star = y_list[-1] + np.random.normal(scale=step_size)
    
            # Metropolis-Hastings condition
            if np.random.rand() < p(x_star, y_star) / p(x_list[-1], y_list[-1]):
                x_curr, y_curr = x_star, y_star
            else:
                x_curr, y_curr = x_list[-1], y_list[-1]

            # update samples list
            x_list.append(x_curr)
            y_list.append(y_curr)
            
        # update the step-size
        step_size_star = random.choice(step_sizes)
        if np.random.rand() < np.random.normal(scale=step_size_star) / np.random.normal(scale=step_size):
            step_size = step_size_star
            j += 1
            update_record.append((j, step_size))
    
    # convert lists to np array for plot environment
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    
    return x_array, y_array, update_record
