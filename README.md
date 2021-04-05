# Finite elements for 2D Problems

## The Problem

Given a polygonal domain $\Omega \subset \mathbb R^2$ with boundary $\partial \Omega = \Gamma_D \cup \Gamma_N$ where $\Gamma_D \cap \Gamma_N = \emptyset$; let $f \in L^2(\Omega)$, $g \in L^2(\Gamma_N)$, $u_D \in L^2(\Gamma_D)$, and $c \in \mathbb R$; the problem is to find $u$ such that
$$
  -\nabla^2 u + c u = f, \quad x \in \Omega
$$$$
  \left. u \right|_{x \in \Gamma_D} = u_D
$$$$
  \left.\frac{\partial u}{\partial n}\right|_{x \in \Gamma_N} = g
$$

See the docs folder for formal documentation on how the solution is computed.

## Current Implementation

The problem is currently set up for $H^1(\Omega)$ problems, where the mesh elements and nodes are prespecified. Mass and stiffness matrices are computed using Sympy operations to build an integration on the reference element then apply this with affine transformations accross all other elements.  To save on computation costs, the load vector is computed through its interpolant rather than direct integration and similarly for the traction vector.  The current linear functionals on the reference element are point evaluation at its vertices.

## Todos and Future Improvements

- Implement elements including normal derivative computations
- Delinearize load vector and traction vector computations
- Sympy computation of basis functions from linear functionals
- External mesh generation and loading
- Square elements
- Implementation of Bartel's red_refine function
