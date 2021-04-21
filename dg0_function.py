#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:14:34 2021

@author: jerome

Discretization for domain using L^\infty functions

Given a set of elements \{K_l\} 
each with vertices \{z_{k_{il}}\} for i = 0, 1, 2
the basis functions are 
\{1(x \in K_l)\}
"""

import numpy as np
import sympy as sm
import scipy.sparse as sp

from discretization import FEMDiscretization
from mesh import Mesh

# symbols for matrix with elements [a, b, c, d]
# global variable
a, b, c, d = sm.symbols("a b c d")
x, y = sm.symbols("x y")
z, w = sm.symbols("z w")


        
        
class DG0(FEMDiscretization):
    """
    Finite element discretization on discontinous,
    piecewise constant elements
    Basis for L^\infty
    """
        
    def __init__(self, mesh=None, is_sparse=True):
        
        # standard initialization, sets mesh and sparsity flag
        super().__init__(mesh=mesh, is_sparse=is_sparse)
        
        self.whole_mass_matrix = None
        self.whole_load_vector = None
        self.whole_traction_vector = None
        
        # phi_hat is the basis function for the reference element \hat K
        # phi_hat = 1(x \in \hat K)
        self.phi_hat = 1
        
        # the ij entry of this is integral of phi_i * phi_j on K-hat
        self.template_matrix = sm.Matrix([[a, b], [c, d]])
        self.integral_phis = self.K_hat_area * np.power(self.phi_hat, 2)
        
    def assemble_matrices(self):
        """
        Build the mass matrix
        for this discretization, only the mass matrix is defined

        Returns
        -------
        M : N x N matrix (possibly sparse)
            Mass matrix.

        """
        
        # data about the mesh, converted to format for quick computations
        # each value is 2 * |K_i| = |K_i| / |\hat K|
        det_B = np.abs(self.mesh.affine_trans_det())
        
        # the data
        # area of |\hat K| cancels
        data_mass = self.integral_phis * det_B  
        
        # assemble matrix, diagonal matrix
        # \int \phi_i \phi_j = |K_j| \delta_{ij}
        offsets = [0]
        M = sp.dia_matrix((data_mass, offsets), 
                          shape=(self.mesh.num_elements, 
                                 self.mesh.num_elements))
        
        if not self.is_sparse:
            M = M.toarray()
            
        self.whole_mass_matrix = M
        
        return M
    
    def interpolate(self, f):
        """
        Interpolate a function onto this DG0 space

        Parameters
        ----------
        f : callable signature f(x) = f([x, y]), vectorized 
            function to interpolate.

        Returns
        -------
        f_vec : array of floats
            (f_vec)_i = f(barycenter of element i).

        """
        barycenters = self.mesh.compute_barycenters().T
        f_vec = f(barycenters)
        
        return f_vec
        

    def project(self, f):
        """
        Do an approximate projection, assuming f is piecewise constant

        Parameters
        ----------
        f : callable, signature f(x) = f([x, y])
            function to project.

        Returns
        -------
        interp : array of floats
            projected values, which is the same as the interpolant here.

        """
        # assuming f is piecewise constant
        # therefore f is equal to its interpolant
        interp = self.interpolate(f)
        
        # P(I(f)) = I(f), projecting the interpolant does nothing
        return interp
        
        
        
    def assemble_load_vector(self, f):
        """
        Build the load vector for the global system
        replaces f by its interpolant for computational efficiency

        Parameters
        ----------
        f : callable, signature f(x)
            forcing function for elliptic problem.

        Returns
        -------
        load_vec : array of floats
            load vector for matrix formulation.

        """
        self.assemble_matrices()
        # using the load vector as M @ f
        
        # build interpolant
        f_vec = self.interpolate(f)
        
        load_vec = self.whole_mass_matrix @ f_vec
                
        self.whole_load_vector = load_vec
        return load_vec
    
    def assemble_traction_vector(self, g):
        """
        Assemble traction vector

        Parameters
        ----------
        g : callable, signature g(x)
            neumann boundary condition.

        Returns
        -------
        traction : array of floats
            traction vector for matrix formulation.

        """
        N = self.mesh.num_elements
        Nn = np.shape(self.neumannbc)[0]
        traction = np.zeros(N)
        
        for edge_num in range(Nn):
            
            # what elements touch this edge? There is only 1
            # extract the element
            
            # relative indices of edge nodes indicate that of phi
            first_node = self.neumannbc[edge_num, 0]
            second_node = self.neumannbc[edge_num, 1]
            
            # viable elements
            viable_element_numbers = np.arange(self.mesh.num_elements)[
                self.__collect_support__(first_node)]
            viable_elements = self.mesh.elements[viable_element_numbers, :]
            
            # which one contains both
            index = np.where(viable_elements == second_node)[0][0]
            the_element_number = viable_element_numbers[index]
                        
            # traction addition
            midpoint = 0.5 * (self.mesh.coordinates[first_node, :] + 
                              self.mesh.coordinates[second_node, :])
            edge_length = np.linalg.norm(self.mesh.coordinates[first_node, :] - 
                                         self.mesh.coordinates[second_node, :])
            
            traction[the_element_number] += edge_length * g(midpoint)
                
        self.whole_traction_vector = traction
                        
        return traction
        
    
if __name__ == "__main__":
    
    # tests
    f = lambda x: np.power(x[0], 2)
    
    mesh = Mesh(coords_fname="meshes/coords_unit_square.dat",
                elements_fname="meshes/elements_unit_square.dat")
    
    dirichlet = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    neumann = np.array([])
    
    dg0 = DG0(mesh=mesh)
    dg0.set_boundaries(dirichlet, neumann)
    
    #for j in range(3): dg0.blue_refine()
    
    
    dg0.plot_mesh()
    
    b = dg0.assemble_load_vector(f)
    print("load vector")
    print(b)
    
    interp = dg0.interpolate(f)
    print("interpolant")
    print(interp)
    
    pf = dg0.project(f)
    print("approx projection")
    print(pf)
