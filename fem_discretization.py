"""
Discretization class for PDE problems using FEM

FEM_Discretization class used to build a Finite element (FE) 
discretization.  Requires mesh.py

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import sympy as sm

import mesh as msh

x, y = sm.symbols("x y")
b00, b01, b10, b11 = sm.symbols("a b c d")

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
            1 - x - y, 
            x, 
            y]
        
        self.grad_phi_hat = np.array([[sm.diff(phi, x), sm.diff(phi, y)] 
                             for phi in self.phi_hat])
        
        self.K_hat_area = 0.5
        
        # the ij entry of this is integral of phi_i * phi_j on K-hat
        
        self.template_matrix = sm.Matrix([[b00, b01], [b10, b11]])
        self.integral_phis = np.zeros([len(self.phi_hat), len(self.phi_hat)])
        self.integral_grad_phis = []
        for i in range(len(self.phi_hat)):
            self.integral_phis[i, :] = [float(sm.integrate(
                self.phi_hat[i] * phi, (y, 0, 1 - x), (x, 0, 1)))
                for phi in self.phi_hat]
            
            fcts = [sm.Matrix(self.grad_phi_hat[i]).T * 
                    self.template_matrix * self.template_matrix.T *
                    sm.Matrix(phi) for phi in self.grad_phi_hat]
            self.integral_grad_phis.append([
                sm.integrate(f[0], (y, 0, 1 - x), (x, 0, 1))
                for f in fcts])
                        
        self.integral_grad_phis = np.array(self.integral_grad_phis)
            
            
        
        #self.__elements_to_list__()
        


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
        
        elements_supporting_both = (elements_supporting_1 + 
                                    elements_supporting_2) == 2
        
        
        return elements_supporting_both
    
    def __find_elements_containing_edge__(self, edge_num):
        """
        Given an edge, what elements contain that edge (for neumann bc)

        Parameters
        ----------
        edge_num : int >= 0
            index of the edge number in the neumann bc.

        Returns
        -------
        usable_elements : array of ints >= 0
            array of indices of the elements containing the edge.

        """
        # what elements contain the first index of the edge
        first_element = self.neumannbc[edge_num, 0]
        possible_elements_indices = np.where(self.mesh.elements == first_element)[0]
        
        # which of these elements also contain the second index of the edge
        second_element = self.neumannbc[edge_num][1]
        usable_elements = np.where(self.mesh.elements[possible_elements_indices, :] == 
                                   second_element)[0]
        
        return usable_elements
    
    def __extract_free_indices__(self):
        """
        Determine the free indices

        Returns
        -------
        free_indices : array of booleans
            selection arrray for the free indices of the matrix formulation.
        dirichlet_indices : array of booleans
            the opposite array.

        """
        dirichlet_elements = np.unique(self.dirichletbc)
        all_elements = np.arange(np.shape(self.mesh.coordinates)[0])
        free_indices = np.isin(all_elements, dirichlet_elements, invert=True)
        dirichlet_indices = np.isin(all_elements, dirichlet_elements)
        
        return free_indices, dirichlet_indices
    
    def __setup_linear_system__(self, c, f, g, u_D):
        """
        Setup a linear system

        Parameters
        ----------
        c : float
            weighting of stiffness term.
        f : callable signature f(x)
            forcing term for elliptic problem.
        g : callable signature g(x)
            neumann boundary condition.
        u_D : callable signature u_D(x)
            dirichlet boundary condition.

        Returns
        -------
        A : nf x nf array (possibly sparse)
            matrix encapsulating mass and stiffness terms on free nodes.
        b : nf long array
            resulting right hand side from matrix formulation.

        """
        self.assemble_matrices()
        #self.assemble_mass_matrix()
        #self.assemble_stiffness_matrix()
        self.assemble_load_vector(f)
        self.assemble_traction_vector(g)
        
        free, taken = self.__extract_free_indices__()
        
        dir_bc_vec = np.array(list(map(u_D, list(self.mesh.coordinates))))
        
        
        A = self.whole_stiffness_matrix[free, :][:, free] + \
            c * self.whole_mass_matrix[free, :][:, free]
            
        
        b = self.whole_load_vector[free] + \
            self.whole_traction_vector[free] - \
            (self.whole_stiffness_matrix[free, :][:, taken] + 
             c * self.whole_mass_matrix[free, :][:, taken]) @ \
                dir_bc_vec[taken]
                
        return A, b
    
    def solve(self, c, f, g, u_D, build_whole=True):
        """
        Solve linear system for finite elements

        Parameters
        ----------
        c : float
            weighting of stiffness term.
        f : callable signature f(x)
            forcing function.
        g : callable signature g(x)
            neumann boundary condition.
        u_D : callable signature u_D(x)
            dirichlet boundary condition.

        Returns
        -------
        res : array
            solution to matrix formulation of elliptic problem.

        """
        # setup the linear system
        A, b = self.__setup_linear_system__(c, f, g, u_D)
        
        # solve
        if self.is_sparse:
            res = spsolve(A, b)
            
        else:
            res = np.linalg.solve(A, b)
            
        if build_whole:
            res = self.__pad_soln__(res, u_D)
        return res
    
    def __pad_soln__(self, u_free, u_D):
        """
        Pad the solution computed on free indices with the boundary conditions

        Parameters
        ----------
        u_free : array
            computed free solutions.
        u_D : callable signature u_D(x)
            dirichlet boundary condition.

        Returns
        -------
        u_whole : array
            solution of u at each node.

        """
        # what goes where
        free, taken = self.__extract_free_indices__()
        
        # assign those known from bc
        u_whole = np.array(list(map(u_D, list(self.mesh.coordinates))))
        
        # assign free indices
        u_whole[free] = u_free
        
        return u_whole  
        
        
    def assemble_mass_matrix(self):
        """
        Assemble mass matrix - wrapper deals with sparsity

        Returns
        -------
        M : N x N matrix (possible sparse)
            Mass matrix for system.

        """
        
        M = self.assemble_mass_matrix_sparse()
        
        if not self.is_sparse:
            M = M.toarray()
            
        self.mass_matrix = M
            
        return M
    
    def assemble_matrices(self):
        
        B_inverses = self.mesh.affine_trans_mat_inverse()
        det_B = self.mesh.affine_trans_det()
        
        data_mass = []
        data_stiff = []
        
        rows = []
        cols = []
        
        num_nodes = np.shape(self.mesh.coordinates)[0]
        grid = np.zeros([num_nodes, num_nodes]) * np.nan
        
        for i in range(num_nodes):
            supp = self.__collect_support__(i)
            elements_to_consider = self.mesh.elements[supp, :]
            
            tmp = np.arange(self.mesh.num_elements)[supp]
            
            for element_num in range(len(elements_to_consider)):
                element = elements_to_consider[element_num]
                
                element_index = tmp[element_num]
                rel_index_i = np.where(element == i)[0][0]
                for j in range(3):
                    rows.append(i)
                    cols.append(element[j])
                    data_mass.append(self.integral_phis[rel_index_i, j] * 
                                     det_B[element_index])
                    data_stiff.append(float(self.integral_grad_phis[
                        rel_index_i, j].subs([
                            (b00, B_inverses[element_index, 0, 0]), 
                            (b01, B_inverses[element_index, 0, 1]),
                            (b10, B_inverses[element_index, 1, 0]),
                            (b11, B_inverses[element_index, 1, 1])])) * 
                            det_B[element_index])
                
        
        M = sp.csr_matrix((data_mass, (rows, cols)), 
                          shape=(num_nodes, num_nodes))
        S = sp.csr_matrix((data_stiff, (rows, cols)), 
                          shape=(num_nodes, num_nodes))
        
        # ensure symmetry
        # M += M.T
        # S += S.T
        
        if not self.is_sparse:
            M = M.toarray()
            S = S.toarray()
            
        self.whole_mass_matrix = M
        self.whole_stiffness_matrix = S
        
        return M, S
                        
                        
                        
                        
                    
                        
                    
            
            
            
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
            for j in range(i, N):
                
                # for two nodes, where do their basis functions overlap in 
                # support
                if i == j:
                    elements_list = self.__collect_support__(i)
                    elements = np.arange(self.mesh.num_elements)[elements_list]
                else:
                    elements_list = self.__collect_support_overlap__(i, j)
                    elements = np.arange(self.mesh.num_elements)[elements_list]
                    
                # for each element, perform integration
                if len(elements) > 0:
                    val = 0
                    rows.append(i)
                    cols.append(j)
                    for m in elements:
                        # which phi_i are we using
                        relative_index_i = np.where(
                            self.mesh.elements[m, :] == i)[0][0]
                        relative_index_j = np.where(
                            self.mesh.elements[m, :] == j)[0][0]
                        
                        # build integrand
                        # integrand = np.dot(
                        #     B_inverses[m, :, :].T @ 
                        #         self.grad_phi_hat[relative_index_i, :], 
                        #     B_inverses[m, :, :].T @ 
                        #         self.grad_phi_hat[relative_index_j, :])
                        integrand = float(self.integral_grad_phis[
                            relative_index_i,relative_index_j].subs([
                                            (b00, B_inverses[m, 0, 0]),
                                            (b01, B_inverses[m, 0, 1]),
                                            (b10, B_inverses[m, 1, 0]),
                                            (b11, B_inverses[m, 1, 1])]))
                        
                        # determine integral and add to entry
                        val += det_B[m] * integrand
                            
                    if i == j:
                        # scaling with diagonals when applying symmetry
                        val /= 2 
                        
                    data.append(float(val))
        
        # build sparse matrix
        mass_matrix = sp.csr_matrix((data, (rows, cols)), shape=[N, N])  
        # ensure is symmetric
        mass_matrix = mass_matrix + mass_matrix.T
                        
        self.whole_mass_matrix = mass_matrix
        return mass_matrix
                


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
            for j in range(i, N):
                
                if i == j:
                    elements_list = self.__collect_support__(i)
                    elements = np.arange(self.mesh.num_elements)[elements_list]
                else:
                    elements_list = self.__collect_support_overlap__(i, j)
                    elements = np.arange(self.mesh.num_elements)[elements_list]
                
                if len(elements) > 0:
                    val = 0
                    rows.append(i)
                    cols.append(j)
                    for m in elements:
                        # which phi_i are we using
                        relative_index_i = np.where(self.mesh.elements[m, :] == i)[0][0]
                        relative_index_j = np.where(self.mesh.elements[m, :] == j)[0][0]
                        
                        # integration is already done, use lookup table
                        val += det_B[m] * \
                            self.integral_phis[relative_index_i, relative_index_j]
                            
                    if i == j:
                        # scaling for applying symmetry
                        val /= 2
                    data.append(val)
        
        # build stiffness matrix and ensure symmetry
        stiffness_matrix = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
        stiffness_matrix = stiffness_matrix + stiffness_matrix.T
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
        stiffness_matrix = self.assemble_stiffness_matrix_sparse()
        
        if not self.is_sparse:
            stiffness_matrix = stiffness_matrix.toarray()
            
        self.whole_stiffness_matrix = stiffness_matrix
        return stiffness_matrix
        
    def assemble_load_vector(self, f):
        """
        Build the load vector for the global system

        Parameters
        ----------
        f : callable, signature f(x)
            forcing function for elliptic problem.

        Returns
        -------
        load_vec : array of floats
            load vector for matrix formulation.

        """
        # recognizing the load vector is S @ f
        # where S is stiffness matrix
        # and f_i = f(z_i)
        if self.whole_mass_matrix is None:
            self.assemble_matrices()
            
        f_vec = np.array(list(map(f, list(self.mesh.coordinates))))
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
        N = np.shape(self.mesh.coordinates)[0]
        Nn = np.shape(self.neumannbc)[0]
        traction = np.zeros(N)
        
        for edge_num in range(Nn):
            
            # what elements touch this edge there is only 1
            needed_element = self.__find_elements_containing_edge__(edge_num)
            # extract the element
            element = needed_element[0]
            
            # relative indices of edge nodes indicate that of phi
            first_node = self.neumannbc[edge_num, 0]
            second_node = self.neumannbc[edge_num, 1]
            
            # extract coordinates for ease of use
            first_coord = self.mesh.coordinates[first_node, :]
            second_coord = self.mesh.coordinates[second_node, :]
            
            # integral scaling
            scaling = np.linalg.norm(self.mesh.coordinates[first_node] - 
                                     self.mesh.coordinates[second_node])
            
            # integration done via linear approximation
            traction[first_node] += scaling * \
                (2 * g(first_coord) + g(second_coord)) / 6
            traction[second_node] += scaling * \
                (g(first_coord) + 2 * g(second_coord)) / 6
                
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
    
    import time
    mesh = msh.Mesh(coords_fname="coordinates2.dat", 
                    elements_fname="elements2.dat")
    
    mesh.plot_mesh(show_indices=True)
    
    r, t, b = sm.symbols("r t b")
    
    step_r = sm.Piecewise((1, r <= 1), (0, r > 1))
    
    theta = sm.atan2(y, x) + sm.Piecewise((2 * sm.pi, y < 0), (0, y >=0 ))
    beta = 2. / 3
    
    
    u_D_full_sm = (1 - r**2) * r**b * sm.sin(b * t)
    f_full_sm = sm.diff(r * sm.diff(u_D_full_sm, r), r) / r + \
        sm.diff(u_D_full_sm, (t, 2)) / r**2
    
    u_D_full_sm = u_D_full_sm.subs(b, beta).simplify()
    f_full_sm = f_full_sm.subs(b, beta).simplify()
    
    u_D_sm = u_D_full_sm * step_r
    f_sm = f_full_sm * step_r
    print(f_full_sm)
    
    u_D_sm = u_D_sm.subs(r, sm.sqrt(x**2 + y**2))
    f_sm = f_sm.subs(r, sm.sqrt(x**2 + y**2))
    
    u_D_sm = u_D_sm.subs(t, theta).simplify()
    f_sm = f_sm.subs(t, theta).simplify()
    
    print(u_D_sm)
    u_D = sm.lambdify([(x, y)], u_D_sm)
    f = sm.lambdify([(x, y)], -f_sm)
    g = lambda x: 0
    
    c = 0
    
    
    disc1 = P1FEMDiscretization(mesh=mesh)
    
    dirichlet_edges = np.array([
        [0, 1],
        [1, 4],
        [4, 5],
        [5, 10], 
        [10, 9],
        [9, 8],
        [8, 3],
        [3, 0]])
    # dirichlet_edges -= 1
    neumann_edges = np.array([])
    
    disc1.set_boundaries(dirichlet_edges, neumann_edges)
    
    for time in range(5):
        disc1.unif_refine()
    
    M, S = disc1.assemble_matrices()
    print(24 * M.toarray())
    print(S.toarray())
    
    
    
    
    
    
    
    
    #disc1.plot_mesh(show_indices=True)
    
    # f = lambda x: 1
    # u_D = lambda x: 0
    # g = lambda x: 0
    # c = 0
    u_sol = disc1.solve(c, f, g, u_D)
    
            
    
    
                    
    
    # for j in range(1):
    #     start = time.time()
    #     disc1.unif_refine()
    #     stop = time.time()
    #     print("refine no. ", j, "time: ", stop - start)
    
    
    # disc1.plot_mesh(show_indices=False)
    
    
    # start = time.time()
    # sol_vec = disc1.solve(c, f, g, u_D)
    # stop = time.time()
    # print(stop - start)
    
    from mpl_toolkits.mplot3d import Axes3D    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    collec = ax.plot_trisurf(disc1.mesh.coordinates[:, 0], 
                              disc1.mesh.coordinates[:, 1], u_sol, 
                              linewidth=0.2)
    ax.view_init(30, 250)
    cbar = fig.colorbar(collec)
    
