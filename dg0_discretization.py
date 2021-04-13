#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:54:50 2021

@author: jerome

Important notes:
    
As a DG0 space, each Phi_j is piecewise constant
As a result, gradients no longer make sense.
Therefore, this space is most useful for 
projections and interpolation.

This space is not useful for solve differential equations.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import sympy as sm


import mesh as msh

x, y = sm.symbols("x y")
b00, b01, b10, b11 = sm.symbols("a b c d")
z0, z1 = sm.symbols("z0, z1")

class DG0FEMDiscretization():
    
    """
    Finite element discretization on discontinous,
    piecewise constant elements
    """
    
    def __init__(self, mesh=None, is_sparse=True):
        self.mesh = mesh
        self.dirichletbc = None
        self.neumannbc = None
        
        self.whole_mass_matrix = None
        self.whole_load_vector = None
        self.whole_traction_vector = None
        
        self.is_sparse = is_sparse 
        
        self.phi_hat = 1
        
        
        self.K_hat_area = 0.5
        
        # the ij entry of this is integral of phi_i * phi_j on K-hat
        
        self.template_matrix = sm.Matrix([[b00, b01], [b10, b11]])
        self.integral_phis = self.K_hat_area * np.power(self.phi_hat, 2)
            
        
    def __collect_support__(self, node_num):
        """
        Determine which elements contain a given node

        Parameters
        ----------
        node_num : int >= 0
            index of the node in question.

        Returns
        -------
        elements_containing_node : numpy array of type int
            Which elements contain the node (which rows of elements array).

        """
        # determine which elements contain the node number given, 
        # take only rows
        #elements_containing_node = np.sum(
        #    self.mesh.elements - node_num == 0, axis=1)
        #elements_containing_node = np.where(
        #    self.mesh.elements == node_num)[0]
        
        elements_containing_node = [int(node_num == lst[0]) + 
                                    int(node_num == lst[1]) + 
                                    int(node_num == lst[2]) for 
                                    lst in list(self.mesh.elements)]
        
        #print(elements_containing_node)
        
        return np.array(elements_containing_node).astype(np.bool)
    
    
    def set_boundaries(self, dirichlet, neumann):
        """
        Set boundary locations

        Input:
            diriclet : array
                locations of dirichlet nodes
            neumann : array
                locations of neumann nodes
        """

        self.dirichletbc = dirichlet
        self.neumannbc = neumann

    def set_mesh(self, mesh):
        self.mesh = mesh
        
        
    def assemble_matrices(self):
        """
        Build the mass matrix

        Returns
        -------
        M : N x N matrix (possibly sparse)
            Mass matrix.

        """
        
        # data about the mesh, converted to format for quick computations
        det_B = np.abs(self.mesh.affine_trans_det())
        
        # the data
        data_mass = self.integral_phis * det_B  
        
        # assemble matrix, diagonal
        offsets = [0]
        M = sp.dia_matrix((data_mass, offsets), 
                          shape=(self.mesh.num_elements, 
                                 self.mesh.num_elements))
        
        if not self.is_sparse:
            M = M.toarray()
            
        self.whole_mass_matrix = M
        
        return M
    
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
        # using the load vector as M @ f
        # where M is mass matrix
        # and f_i = f(z_i)
        self.assemble_matrices()
        
        barycenters = self.mesh.compute_barycenters()
        f_vec = np.array(list(map(f, list(barycenters))))
        
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
    
    def unif_refine(self):
        """
        Uniformly refine mesh and boundaries by
        cutting each triangle into 4 congruent triangles
        """

        # refine mesh
        #midpoints, nodes_have_encountered = self.mesh.unif_refine()
        midpoints, nodes_have_encountered = self.mesh.blue_refine()
        #midpoints, nodes_have_encountered = self.mesh.pink_refine()

        # update boundary conditions
        new_dirichlet = np.zeros([2 * np.shape(self.dirichletbc)[0], 2])
        for j in range(np.shape(self.dirichletbc)[0]):
            current_edge = self.dirichletbc[j, :]
            new_dirichlet[2 * j, :] = [current_edge[0], 
                                       midpoints[current_edge[0], 
                                                 current_edge[1]]]
            new_dirichlet[2 * j + 1, :] = [midpoints[current_edge[0], 
                                                     current_edge[1]], 
                                           current_edge[-1]]

        new_neumann = np.zeros([2 * np.shape(self.neumannbc)[0], 2])
        for j in range(np.shape(self.neumannbc)[0]):
            current_edge = self.neumannbc[j, :]
            new_neumann[2 * j, :] = [current_edge[0], 
                                     midpoints[current_edge[0], 
                                               current_edge[1]]]
            new_neumann[2 * j + 1, :] = [midpoints[current_edge[0], 
                                                   current_edge[1]], 
                                         current_edge[-1]]

        self.set_boundaries(new_dirichlet.astype(np.int), 
                            new_neumann.astype(np.int))

    def refine_edges(self, edges, midpoints):
        """
        Helper function to refine boundaries

        Parameters
        ----------
        edges : N x 2 array
            edges in question.
        midpoints : N x N sparse array
            indicates where each midpoint is indexed.

        Returns
        -------
        new_edges : 2 N x 2 array
            new boundary edges after refinement.

        """
        if len(edges) == 0:
            return np.array([])
        else:
            new_edges = np.zeros([2 * len(edges), 2]).astype(np.int)
            
            new_edges[::2, 0] = edges[:, 0]
            new_edges[1::2, 1] = edges[:, 1]
            
            new_edge_indices = [midpoints[e[0], e[1]] 
                                     for e in list(edges)]
            new_edges[1::2, 0] = np.array(new_edge_indices)
            new_edges[::2, 1] = np.array(new_edge_indices)
            
            return new_edges
    
    def blue_refine(self, memory_intensive=False):
        """
        A faster uniform refinement

        Parameters
        ----------
        memory_intensive : boolean
            indicates if mesh is high resolution and will 
            require a lot of memory
            If this is True, relies on the slower pink_refine.
            The default is False
        Returns
        -------
        None.

        """
        if memory_intensive:
            midpoints, have_encountered = self.mesh.pink_refine()
        else:
            midpoints, have_encountered = self.mesh.blue_refine()
            
        self.dirichletbc = self.refine_edges(self.dirichletbc, midpoints)
        self.neumannbc = self.refine_edges(self.neumannbc, midpoints)
    
    def plot_mesh(self, show_indices=False, **kwargs):
        """
        Plot mesh with boundary conditions

        Parameters
        ----------
        show_indices : boolean, optional
            whether to display index numbers for each node.
            The default is False.
        **kwargs : extra arguments for plotting

        Returns
        -------
        fig : matplotlib figure
            figure containing plot.
        ax : matplotlib axis
            axis on figure containing plot.

        """
        
        # plot mesh showing indices
        fig, ax = self.mesh.plot_mesh(show_indices=show_indices, **kwargs)
        
        # colors of edges
        dirichlet_color = "g"
        neumann_color = "y"

        # plot dirichlet bc
        for edge_no in range(np.shape(self.dirichletbc)[0]):
            
            x_lst = self.mesh.coordinates[self.dirichletbc[edge_no], 0]
            y_lst = self.mesh.coordinates[self.dirichletbc[edge_no], 1]
            
            if edge_no == 0:
                ax.plot(x_lst, y_lst, dirichlet_color, label="dirichlet")
            else:
                ax.plot(x_lst, y_lst, dirichlet_color)
        
        # plot neumann bc
        for edge_no in range(np.shape(self.neumannbc)[0]):
            
            x_lst = self.mesh.coordinates[self.neumannbc[edge_no], 0]
            y_lst = self.mesh.coordinates[self.neumannbc[edge_no], 1]
            
            if edge_no == 0:
                ax.plot(x_lst, y_lst, neumann_color, label="neumann")
            else:
                ax.plot(x_lst, y_lst, neumann_color)
            
        # add legend and return
        ax.legend()
        return fig, ax
    
    
if __name__ == "__main__":
    
    import time
    mesh = msh.Mesh(coords_fname="meshes/coords_unit_square.dat", 
                    elements_fname="meshes/elements_unit_square.dat")
    
    
    f = lambda x: x[0] * x[1]
    g = lambda x: 1
    
    dirichlet_edges = np.array([
        [0, 1], 
        [1, 2], 
        [2, 3]])
    # dirichlet_edges -= 1
    neumann_edges = np.array([[3, 0]])
    
    disc0 = DG0FEMDiscretization(mesh=mesh)
    disc0.set_boundaries(dirichlet_edges, neumann_edges)
    
    disc0.blue_refine()
    disc0.blue_refine()

    
    load_vector = disc0.assemble_load_vector(f)
    traction_vector = disc0.assemble_traction_vector(g)
    
    print(load_vector)