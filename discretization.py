#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:22:40 2021

@author: jerome

Abstract finite element discretization
Useful to keep these book-keeping methods from jumbling up everything else
"""

import numpy as np

class FEMDiscretization():
    """
    Abstract class for finite element discretization
    """
    
    def __init__(self, mesh=None, is_sparse=True):
        self.mesh = mesh
        self.dirichletbc = None
        self.neumannbc = None
        
        self.is_sparse = is_sparse 
        
        # area of reference element
        self.K_hat_area = 0.5
        
    def set_mesh(self, mesh):
        self.mesh = mesh
        
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