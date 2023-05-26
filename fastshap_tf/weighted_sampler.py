import math
from decimal import Decimal
import numpy as np

def beta_constant(a, b): # using Decimal module to deal with underflow of values
   '''
   the second argument (b; beta) should be integer in this function
   '''
   beta_fct_value=Decimal(1/a)
   for i in range(1,b):
        beta_fct_value=beta_fct_value*Decimal((i/(a+i)))
   return beta_fct_value


def w(M,a,b,j):
    return (M*beta_constant(j+b-1,M-j+a)/beta_constant(a,b))


def w_tilde(M,a,b,j): # use exponent based representation for handling 
    '''
    - w_tilde(j,a,b) = bin(M-1,j-1) * w(j,a,b) = T1 * T2
      We can do this approximation: express both T1 and T2 in the form: <a.bcdef x 10^{g}>
      Now, T1*T2 = (a1.b1c1d1e1f1 * a2.b2c2d2e2f2) * 10^{g1+g2}
    - We use the Decimal module to do this
    '''
    # exp_bc, base_bc = get_in_exp_form(binom_coeff(M-1,j-1))
    # exp_w, base_w = get_in_exp_form(w(M,a,b,j))
    #return ((base_bc*base_w)*math.pow(10,exp_bc+exp_w))
    return Decimal(math.comb(M-1,j-1))*Decimal(w(M,a,b,j))


def beta_shapley_subset_cardinality_wt(N,a,b,j): # for the subsets with cardinality j, this function gives the weight
    return w_tilde(N,a,b,j)/(j*(N-j))


def compute_weighted_shapley_wts(m, alpha=16, beta=1):
    weight_list=np.zeros(m)
    for j in range(1,m):
        weight_list[j-1] = beta_shapley_subset_cardinality_wt(m,alpha,beta,j)
    return weight_list