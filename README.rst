Forked from tensorly/tensorly to implement an optimization submodule, check tensorly/contrib/optimization

# Optimization modules for tensorly

The present fork implements a set of methods for deeper optimization persepctives using Tensorly. 

Using a Parafac class, it offers the possibility to compute constrained decompositions, fixing some parameters, easily chosing the initial parameters and so on. 

As of 11/02/2019, implemented methods are projected accelerated gradient, hierarchical ALS (for nonnegativity only), multiplicative updates and plain ALS. AOADMM is on its way.

TODO :
- Write a documentation
- Fix unfoldings being computed at each iterations
- Add projection API
- Add other models (Tucker, Parafac2 etc)
- Merge into tensorly ?
