# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 23:58:04 2026

@author: khate pontay
"""

import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

#POSTERIOR
import scipy as sp

unnormalized_posterior = likelihood_out * uniform_dist
plt.plot(mu, unnormalized_posterior)
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.show()