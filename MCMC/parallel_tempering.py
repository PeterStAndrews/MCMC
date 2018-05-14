

# Parallel tempering sampler
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


def Parallel_Tempering(p, step_size, n, e, x0, y0):
    '''samples the target distribution, p, using len(step_size) 
    Markov chains that take n samples before proposing an exchange
    a total of e times
    
    :param::'p': the target distribution:
    :param::'step_size': array of Q sigmas
    :param::'n': length of MH sampling before exchange 
    :param::'e': the number of exchanges
    :param::'x0': the initial x guess 
    :param::'y0': the initial y guess
    '''
    num_chains = len(step_size)
    
    # create distributions for each chain
    Ps = []
    for chain in range(num_chains):
        Ps.append(lambda xi,yi: p(xi,yi))
      
    # randomise the starting points of the chains
    X0=np.random.uniform(low=-5.1, high=-5.0, size=num_chains)
    Y0=np.random.uniform(low=-5.1, high=-5.0, size=num_chains)
    
    # set the value of the first chain
    X0[1]=x0
    Y0[1]=y0
    
    # create lists for x y data
    x=[ [] for i in range(num_chains)]
    y=[ [] for i in range(num_chains)]
    
    for exchange in range(e):
        for chain in range(num_chains):
            
            # generate samples 
            samples = metropolis_hastings(p, n, X0[chain], Y0[chain], step_size[chain])
            
            # format data into lists of x and y
            sample_list = samples.tolist()
            x_new = [x[0] for x in sample_list]
            y_new = [y[1] for y in sample_list]
            
            # update the chain with the new samples
            x[chain] = x[chain] + x_new
            y[chain] = y[chain] + y_new
            
        for chain in range(num_chains-1):
            
            # select the proposal values to exchange
            x1 = x[chain][-1]
            y1 = y[chain][-1]
            x2 = x[chain+1][-1]
            y2 = y[chain+1][-1]
        
            # compute exchange ratio
            E = math.exp((Ps[chain](x2,y2)*Ps[chain+1](x1,y1))/(Ps[chain](x1,y1)*Ps[chain+1](x2,y2))) 
            
            # propose exchange
            if np.random.uniform() < E:
                
                X0[chain] = x2
                X0[chain+1] = x1
                
                Y0[chain] = y2
                Y0[chain+1] = y1
                
    return x, y

def ptarget(x, y):
    '''the distribution we wish to sample'''
    return NotImplementedError