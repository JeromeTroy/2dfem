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
                        
        
        self.integral_grad_phis = sm.lambdify([[b00, b01, b10, b11]],
                                            np.array(self.integral_grad_phis))
        #fcts = [sm.lambdify([b00, b01, b10, b11], f) 
        #        for f in self.integral_grad_phis.ravel()]
        #self.integral_grad_phis = np.reshape(np.array(fcts), (3, 3))
            
            
        
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
       
    
    def assemble_matrices(self):
        """
        Build the mass and stiffness matrices

        Returns
        -------
        M : N x N matrix (possibly sparse)
            Mass matrix.
        S : N x N matrix (possibly sparse)
            Stiffness matrix.

        """
        
        # data about the mesh, converted to format for quick computations
        B_inverses = self.mesh.affine_trans_mat_inverse()
        B_inverses_list = list(map(lambda x: x.ravel(), list(B_inverses)))
        det_B = np.abs(self.mesh.affine_trans_det())
        
        # the data
        data_mass = list(map(lambda x: self.integral_phis * x, list(det_B)))
        data_stiff = list(map(lambda x:
                              np.array(self.integral_grad_phis(x[0])) * x[1], 
                              zip(B_inverses_list, list(det_B))))
        
        # where it will go
        tmp_zip = list(map(lambda x: np.meshgrid(x, x), 
                           list(self.mesh.elements)))
        rows_mat, cols_mat = zip(*tmp_zip)
        
        # put into nice format
        data_mass = np.array(data_mass).ravel()
        data_stiff = np.array(data_stiff).ravel()
        rows = np.array(rows_mat).ravel()
        cols = np.array(cols_mat).ravel()
        
        
        # assemble matrices
        M = sp.csr_matrix((data_mass, (rows, cols)), 
                          shape=(self.mesh.num_nodes, self.mesh.num_nodes))
        S = sp.csr_matrix((data_stiff, (rows, cols)), 
                          shape=(self.mesh.num_nodes, self.mesh.num_nodes))
        
        if not self.is_sparse:
            M = M.toarray()
            S = S.toarray()
            
        self.whole_mass_matrix = M
        self.whole_stiffness_matrix = S
        
        return M, S
        
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
            # extract the element
            
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
        
                                
    
    def error_approx(self, u_sol, u_true):
        
        z0, z1 = sm.symbols('z0, z1')
        
        first_l2 = np.dot(u_sol, self.whole_mass_matrix @ u_sol)
        
        u_squared = u_true(self.template_matrix * sm.Matrix(x, y) + 
                      sm.Matrix([z0, z1]))
        u_l2 = sm.integrate(u_squared, (y, 0, 1 - x), (x, 0, 1))
        u_l2_fun = sm.lambdify([b00, b01, b10, b11, z0, z1], u_l2)
        # TODO: apply to vectorized elements / coordinates
        
        # TODO: build middle term u * phi
        # TOOD: H1 error
        # vectorized values at coordinates
        u_true_vec = np.array(list(map(u_true, list(self.mesh.coordinates))))

        difference = u_sol - u_true_vec
        
        l2_error = np.dot(difference, 
                          self.whole_mass_matrix @ difference)
        h1_error = l2_error + np.dot(difference, 
                                     self.whole_stiffness_matrix @ difference)
        
        return h1_error, l2_error
    
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
    
    num_refine = 8
    l2_errors = np.zeros(num_refine + 1)
    h1_errors = np.zeros(num_refine + 1)
    
    mesh.plot_mesh(show_indices=True)
    
    start = time.time()
    r, t, b = sm.symbols("r t b")
    
    step_r = sm.Piecewise((1, x**2 + y**2 < 1), (0, x**2 + y**2 >= 1))
    
    theta = sm.atan2(y, x) + sm.Piecewise((2 * sm.pi, y < 0), (0, y >=0 ))
    beta = 2. / 3
    
    
    u_D_full_sm = (1 - r**2) * r**b * sm.sin(b * t)
    f_full_sm = sm.diff(r * sm.diff(u_D_full_sm, r), r) / r + \
        sm.diff(u_D_full_sm, (t, 2)) / r**2
    
    u_D_full_sm = u_D_full_sm.subs(b, beta).simplify()
    f_full_sm = f_full_sm.subs(b, beta).simplify()
    
    u_D_full_sm = u_D_full_sm.subs(r, sm.sqrt(x**2 + y**2))
    f_full_sm = f_full_sm.subs(r, sm.sqrt(x**2 + y**2))
    
    u_D_full_sm = u_D_full_sm.subs(t, theta).simplify()
    f_full_sm = f_full_sm.subs(t, theta).simplify()
    
    #print(u_D_sm)
    u_D = sm.lambdify([(x, y)], u_D_full_sm)
    f = sm.lambdify([(x, y)], -f_full_sm)
    
    g = lambda x: 0
    
    stop = time.time()
    
    print("Functions computed, time: ", stop - start)
    
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
    
    #u_sol = disc1.solve(c, f, g, u_D)
    
    #l2_errors[0], h1_errors[0] = disc1.error_approx(u_sol, u_D)
    
    for alpha in range(num_refine):
        print("refine no. ", alpha + 1)
        disc1.blue_refine(memory_intensive=True)
        
        #u_sol = disc1.solve(c, f, g, u_D)
        #l2_errors[alpha + 1], h1_errors[alpha + 1] = disc1.error_approx(u_sol, 
        #                                                                u_D)
    
    #disc1.plot_mesh()
    # l2_order = np.log2(l2_errors[:-1] / l2_errors[1:])
    # h1_order = np.log2(h1_errors[:-1] / h1_errors[1:])
    
    # print("L2 errors: ", l2_errors)
    # print("L2 orders: ", l2_order)
    # print("H1 errors: ", h1_errors)
    # print("H1 orders: ", h1_order)
    
    # from mpl_toolkits.mplot3d import Axes3D    
    
    # fig = plt.figure()
    # cmap = plt.get_cmap('viridis')
    # ax = fig.gca(projection='3d')
    # collec = ax.plot_trisurf(disc1.mesh.coordinates[:, 0], 
    #                           disc1.mesh.coordinates[:, 1], u_sol, 
    #                           linewidth=0.2, antialiased = True,
    #                           edgecolor = 'grey', cmap=cmap)
    # ax.view_init(90, 0)
    # cbar = fig.colorbar(collec)
    
