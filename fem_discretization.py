"""
Discretization class for PDE problems using FEM

FEM_Discretization class used to build a Finite element (FE) 
discretization.  Requires mesh.py

"""

import numpy as np
import matplotlib.pyplot
import scipy.sparse as sp
from scipy.special import roots_legendre
from scipy.integrate import dblquad

import mesh as msh


class P1FEMDiscretization():
    """
    Finite Element discretization for continuous, piecewise linear 
    functions

    Includes mesh (Mesh() entity), and
    dirichletbc, neumannbc: arrays of dirichlet and neumann boundary types
    """

    def __init__(self, mesh=None, is_sparse=True):
        self.mesh = mesh
        self.dirichletbc = None
        self.neumannbc = None
        
        self.whole_mass_matrix = None
        self.whole_stiffness_matrix = None
        self.whole_load_vector = None
        
        self.is_sparse = is_sparse 
        
        self.phi_hat = [
            lambda x, y: -x - y - 1, 
            lambda x, y: x, 
            lambda x, y: y]
        
        self.grad_phi_hat = np.array([
            [-1, -1], 
            [1, 0], 
            [0, 1]
            ])
        
        self.K_hat_area = 0.5
        
        # the ij entry of this is integral of phi_i * phi_j on K-hat
        self.integral_phis = 1 / 24 * np.array([
            [2, 1, 1], 
            [1, 2, 1], 
            [1, 1, 2]
            ])


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
        elements_containing_node = np.where(
            self.mesh.elements == node_num)[0]
        
        return elements_containing_node
        
    def __collect_support_overlap__(self, basis_no_1, basis_no_2):
        """
        Determine where the support overlaps between two basis functions

        Parameters
        ----------
        basis_no_1 : int >= 0
            Index of first basis function.
        basis_no_2 : int >= 0
            Index of second basis function.

        Returns
        -------
        elements_supporting_both : array of type int
            List of elements which are in the support of both functions.

        """
        # main nodes are also basis_no_i
        elements_supporting_1 = self.__collect_support__(basis_no_1)
        elements_supporting_2 = self.__collect_support__(basis_no_2)
        
        elements_supporting_both = np.intersect1d(elements_supporting_1, 
                                                  elements_supporting_2)
        
        return elements_supporting_both
    
    def assemble_mass_matrix(self):
        """
        Assemble mass matrix - wrapper deals with sparsity

        Returns
        -------
        M : N x N matrix (possible sparse)
            Mass matrix for system.

        """
        if self.is_sparse:
            M = self.assemble_mass_matrix_sparse()
        else:
            M = self.assemble_mass_matrix_dense()
            
        return M
    
    
    def assemble_mass_matrix_dense(self):
        """
        Assemble the mass matrix (dense version)

        Returns
        -------
        mass_matrix : N x N array of floats
            mass matrix for the problem.

        """
        # details on affine transformations hold on to these
        B_inverses = self.mesh.affine_trans_mat_inverse()
        det_B = self.mesh.affine_trans_det()
        
        # allocate storage
        N = np.shape(self.mesh.coordinates)[0]
        mass_matrix = np.zeros([N, N])
        
        # element by element
        for i in range(N):
            for j in range(i, N):
                
                # for two nodes, where do their basis functions overlap in 
                # support
                elements_list = self.__collect_support_overlap__(i, j)
                
                # for each element, perform integration
                for m in elements_list:
                    # which phi_i are we using
                    relative_index_i = np.where(self.mesh.elements[m, :] == i)[0][0]
                    relative_index_j = np.where(self.mesh.elements[m, :] == j)[0][0]
                    
                    # build integrand
                    integrand = np.dot(self.grad_phi_hat[relative_index_i, :], 
                                       B_inverses[m] @ B_inverses[m].T @ 
                                       self.grad_phi_hat[relative_index_j, :])
                    
                    # determine integral and add to entry
                    mass_matrix[i, j] += self.K_hat_area * det_B[m] * \
                        integrand
        # is symmetric
        mass_matrix += mass_matrix.T
        np.fill_diagonal(mass_matrix, 0.5 * np.diag(mass_matrix))
        
                        
        self.whole_mass_matrix = mass_matrix
        return mass_matrix
    
    def assemble_mass_matrix_sparse(self):
        """
        Assemble mass matrix (sparse version)

        Returns
        -------
        mass_matrix : N x N sparse matrix
            mass matrix for the system.

        """
        # details on affine transformations hold on to these
        B_inverses = self.mesh.affine_trans_mat_inverse()
        det_B = self.mesh.affine_trans_det()
        
        N = np.shape(self.mesh.coordinates)[0]
        
        data = []
        rows = []
        cols = []
        
        # collect data
        # element by element
        for i in range(N):
            for j in range(N):
                
                # for two nodes, where do their basis functions overlap in 
                # support
                elements_list = self.__collect_support_overlap__(i, j)
                
                # for each element, perform integration
                if len(elements_list) > 0:
                    val = 0
                    rows.append(i)
                    cols.append(j)
                    for m in elements_list:
                        # which phi_i are we using
                        relative_index_i = np.where(
                            self.mesh.elements[m, :] == i)[0][0]
                        relative_index_j = np.where(
                            self.mesh.elements[m, :] == j)[0][0]
                        
                        # build integrand
                        integrand = np.dot(self.grad_phi_hat[relative_index_i, :], 
                                           B_inverses[m] @ B_inverses[m].T @ 
                                           self.grad_phi_hat[relative_index_j, :])
                        
                        # determine integral and add to entry
                        val += self.K_hat_area * det_B[m] * \
                            integrand
                    data.append(val)
        
        # build sparse matrix
        mass_matrix = sp.csr_matrix((data, (rows, cols)), shape=[N, N])        
                        
        self.whole_mass_matrix = mass_matrix
        return mass_matrix
                

        
    def assemble_stiffness_matrix_dense(self):
        """
        Assemble stiffness matrix for system - dense version

        Returns
        -------
        stiffness_matrix : N x N matrix
            Stiffness matrix.

        """
        det_B = self.mesh.affine_trans_det()
        
        N = np.shape(self.mesh.coordinates)[0]
        stiffness_matrix = np.zeros([N, N])
        
        for i in range(N):
            for j in range(N):
                
                element_list = self.__collect_support_overlap__(i, j)
                
                for m in element_list:
                    # which phi_i are we using
                    relative_index_i = np.where(self.mesh.elements[m, :] == i)
                    relative_index_j = np.where(self.mesh.elements[m, :] == j)
                    
                    # integration is already done, use lookup table
                    stiffness_matrix[i, j] += det_B[m] * \
                        self.integral_phis[relative_index_i, relative_index_j]
                        
        self.whole_stiffness_matrix = stiffness_matrix
        
        return stiffness_matrix

    def assemble_stiffness_matrix_sparse(self):
        """
        Assemble stiffness matrix - sparse version

        Returns
        -------
        stiffness_matrix : N x N sparse matrix
            stiffness matrix.

        """
        det_B = self.mesh.affine_trans_det()
        
        N = np.shape(self.mesh.coordinates)[0]
        data = []
        rows = []
        cols = []
        
        for i in range(N):
            for j in range(N):
                
                element_list = self.__collect_support_overlap__(i, j)
                
                if len(element_list) > 0:
                    val = 0
                    rows.append(i)
                    cols.append(j)
                    for m in element_list:
                        # which phi_i are we using
                        relative_index_i = np.where(self.mesh.elements[m, :] == i)[0][0]
                        relative_index_j = np.where(self.mesh.elements[m, :] == j)[0][0]
                        
                        # integration is already done, use lookup table
                        val += det_B[m] * \
                            self.integral_phis[relative_index_i, relative_index_j]
                            
                    data.append(val)
        
        stiffness_matrix = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
        self.whole_stiffness_matrix = stiffness_matrix
        
        return stiffness_matrix
    
    def assemble_stiffness_matrix(self):
        """
        Wrapper deals with sparsity

        Returns
        -------
        stiffness_matrix : N x N matrix (possibly sparse0)
            stiffness matrix.

        """
        if self.is_sparse:
            stiffness_matrix = self.assemble_stiffness_matrix_sparse()
        else:
            stiffness_matrix = self.assemble_stiffness_matrix_dense()
            
        self.whole_stiffness_matrix = stiffness_matrix
        return stiffness_matrix
        
    def assemble_load_vector(self, f):
        
        # recognizing the load vector is S @ f
        # where S is stiffness matrix
        # and f_i = f(z_i)
        if self.whole_stiffness_matrix is None:
            self.assemble_stiffness_matrix()
            
        f_vec = np.array(list(map(f, list(self.mesh.coordinates))))
        load_vec = self.whole_stiffness_matrix @ f_vec
        
        self.whole_load_vector = load_vec
        
        return load_vec
    
    
    def unif_refine(self):
        """
        Uniformly refine mesh and boundaries by
        cutting each triangle into 4 congruent triangles
        """

        # refine mesh
        midpoints, nodes_have_encountered = self.mesh.unif_refine()

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


    def neumann_edge_lengths(self):
        """
        Compute lengths of neumann edges for boundaries
        """

        if self.neumannbc is not None:
            # coordinates of each edge
            first_coords = self.mesh.get_coordinates()[self.neumannbc[:, 0], :]
            second_coords = self.mesh.get_coordinates()[self.neumannbc[:, 1], :]

            edge_lengths = np.sqrt(np.sum(np.power(
                second_coords - first_coords, 2), 1))

            return edge_lengths

        else:
            return None

    def compute_neumann_midpoints(self):
        """
        Compute midpoints of Neumann bc edges
        """

        if self.neumannbc is not None:
            # coordinates of each edge
            first_coords = self.mesh.get_coordinates()[self.neumannbc[:, 0], :]
            second_coords = self.mesh.get_coordinates()[self.neumannbc[:, 1], :]

            midpoints = 0.5 * (first_coords + second_coords)

            return midpoints

        else:
            return None

    def compute_neumann_normals(self):
        """
        Compute normal vectors to neumann bc edges
        """

        if self.neumannbc is not None:
            # coordinates of each edge
            first_coords = self.mesh.get_coordinates()[self.neumannbc[:, 0], :]
            second_coords = self.mesh.get_coordinates()[self.neumannbc[:, 1], :]

            # vectors from each node to the next
            vectors = second_coords - first_coords

            # normal to 2d vector
            normals = vectors[:, ::-1]
            normals[:, -1] *= -1

            return normals

        else:
            return None
        
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
    
    mesh = msh.Mesh(coords_fname="coordinates2.dat", 
                    elements_fname="elements2.dat")
    
    f = lambda x: 1
    
    disc1 = P1FEMDiscretization(mesh=mesh)
    
    dirichlet_edges = np.array([
        [0, 1], 
        [1, 4], 
        [4, 5], 
        [5, 10], 
        [10, 9]])
    neumann_edges = np.array([
        [9, 8], 
        [8, 3],
        [3, 0]])
    
    disc1.set_boundaries(dirichlet_edges, neumann_edges)
    
    disc1.plot_mesh()
    
    disc1.unif_refine()
    
    disc1.plot_mesh()
    
    disc1.unif_refine()
    
    disc1.plot_mesh()
    
