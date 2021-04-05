"""
Mesh class and related methods

Mesh class is the base for finite element (FE)
discretization.  This breaks up a domain into 
triangles.  Currently only designed for 2D domains

"""

import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


class Mesh():
    """
    Mesh for a domain

    Contains arrays for coordinates of nodes on the mesh
    and the elements (triangles) which have vertices
    corresponding to mesh nodes
    """

    def __init__(self, coordinates=None, elements=None, 
                 coords_fname=None, elements_fname=None, is_matlab_like=True):
        
        if coordinates is not None:
            self.set_coordinates(coordinates)
        elif coords_fname is not None:
            self.load_coordinates_from_file(coords_fname)
        
        if elements is not None:
            self.set_elements(elements)
        elif elements_fname is not None:
            self.load_elements_from_file(elements_fname, 
                                         is_matlab_like=is_matlab_like)
            
        # bound the default element
        self.yhat_bound = lambda x: 1 - x
        self.xhat_bound = [0, 1]
        
        self.det_B_vals = None
        self.B_inverses = None
        self.B_matrices = None


    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        # get number of nodes
        self.num_nodes = np.shape(self.coordinates)[0]

    def set_elements(self, elements):
        self.elements = elements
        self.elements = self.elements.astype(int)
        # get number of elements
        self.num_elements = np.shape(self.elements)[0]

    def get_coordinates(self):
        return self.coordinates

    def get_elements(self):
        return self.elements

    def get_unique_edges(self):
        """
        Extract unique edges of a mesh

        Returns
        -------
        unique_edges : list of 2-tuples
            list of unique edges.

        """
        # get all edges
        n = self.num_nodes
        first_edges = np.sort(self.elements[:, (0, 1)], axis=1)
        second_edges = np.sort(self.elements[:, (1, 2)], axis=1)
        third_edges = np.sort(self.elements[:, (0, -1)], axis=1)
        
        all_edges = list(first_edges) + \
            list(second_edges) + list(third_edges)
            
        rows, cols = zip(*all_edges)
        edges_mat = csr_matrix((np.ones(len(rows)), (rows, cols)), 
                               shape=(n, n))
        
        unique_edges = list(zip(*edges_mat.nonzero()))
        return unique_edges
            
        
    def load_coordinates_from_file(self, coord_fname):
        """
        Load coordinates from given file

        Input:
            coord_fname : string
                file name from which to load coordinates
        """

        # open file as read only
        fin = open(coord_fname, "r")

        # read in using numpy
        coordinates = np.loadtxt(fin)
        self.set_coordinates(coordinates)

    def load_elements_from_file(self, elements_fname, is_matlab_like=True):
        """
        Load element array from given file

        Input:
            elements_fname : string
                file name from which to load coordinates
            is_matlab_like : boolean, optional
                indicates if input is like that of matlab, i.e. indexing
                starting at 1, the default is True
        """

        # open file as read only
        fin = open(elements_fname, "r")

        # read in using numpy
        elements = np.loadtxt(fin)
        if is_matlab_like:
            elements -= 1
        self.set_elements(elements)

    def unif_refine(self):
        """
        Apply one level of uniform refinement by cutting each triangle into
        4 congruent triangles
        """

        new_num_elements = 4 * self.num_elements

        # will increment number of nodes
        new_num_nodes = self.num_nodes

        # allocate space for midpoints of each line segment
        midpoints = np.zeros([self.num_nodes, self.num_nodes]).astype(int)

        # keep track of what we have done
        have_encountered = []

        # iterate across each element of the mesh
        for j in range(self.num_elements):
            # triangles have 3 sides, so do this for each side
            for k in range(3):
                # determine indices in midpoints array to use
                i1 = self.elements[j, k]
                i2 = self.elements[j, (k + 1) % 3]

                # have not dealt with this edge before
                if ((i1, i2) not in have_encountered) and ((i2, i1) not in have_encountered):
                    have_encountered.append((i1, i2))

                    # determine midpoint, and note which element this will
                    # correspond to
                    midpoints[i1, i2] = new_num_nodes
                    midpoints[i2, i1] = new_num_nodes
                    
                    # adding another node
                    new_num_nodes += 1
                    

        # now that we have recorded where each node will be,
        # we can add them to the coordinates list
        # and add elements to the elements list

        # allocate space for coordinates
        new_coordinates = np.zeros([new_num_nodes, np.shape(self.coordinates)[-1]])
        # copy over previous
        new_coordinates[:self.num_nodes, :] = np.copy(self.coordinates)

        # add the new nodes
        # each new node was the midpoint of a line segment encountered
        for index in range(len(have_encountered)):
            # indices of each end
            i1, i2 = have_encountered[index]
            # compute midpoint and add to coordinates
            new_coordinates[self.num_nodes + index, :] = 0.5 * (
                self.coordinates[i1, :] + self.coordinates[i2, :]
            )

        # new elements, store 3 entries in each row for the 3 vertices
        new_elements = np.zeros([new_num_elements, 3])
        # iterate over current elements
        for j in range(self.num_elements):
            current_element = self.elements[j, :]

            # store new elements
            new_elements[4 * j, :] = [current_element[0],
                                        midpoints[current_element[0], current_element[1]],
                                        midpoints[current_element[-1], current_element[0]]
                                    ]
            new_elements[4 * j + 1, :] = [current_element[1],
                                            midpoints[current_element[1], current_element[2]],
                                            midpoints[current_element[0], current_element[1]]
                                        ]
            new_elements[4 * j + 2, :] = [current_element[-1],
                                            midpoints[current_element[-1], current_element[0]],
                                            midpoints[current_element[1], current_element[2]]
                                        ]
            new_elements[4 * j + 3, :] = [midpoints[current_element[0], current_element[1]],
                                            midpoints[current_element[1], current_element[2]],
                                            midpoints[current_element[-1], current_element[0]]
                                        ]

        # switch over to new elements and nodes
        self.set_coordinates(new_coordinates)
        self.set_elements(new_elements)

        # return useful tools for other purposes
        return midpoints, have_encountered

    def blue_refine(self):
        """
        A faster uniform refinement

        Returns
        -------
        midpoints_mat : N x N sparse matrix
            matrix indicating positions of each midpoint in new coordinates.
        have_encountered : N long list
            list of edges encountered.

        """
        
        # split each element into 4
        new_num_elements = 4 * self.num_elements
        new_elements = np.zeros([new_num_elements, 3])
        
        new_coordinates = list(np.copy(self.coordinates))
        
        indices = np.arange(3)
        indices_to_lookup = np.zeros(
            int(self.num_nodes ** 2)).astype(np.uint16)
        
        midpoint_indices = [0] * 3
        
        # standard output for refining boundaries
        have_encountered = []
        midpoints_mat = np.zeros([self.num_nodes, 
                                  self.num_nodes]).astype(np.int)
        
        for el_num in range(self.num_elements):
            
            # where to look 
            entries = list(zip(self.elements[el_num, indices], 
                               self.elements[el_num, (indices + 1) % 3]))
            where = [self.num_nodes * np.min(e) + np.max(e) - 1 for 
                     e in entries]
            
            # midpoints for 01, 12, 20
            midpoints = 0.5 * (
                self.coordinates[self.elements[el_num, indices]] + 
                self.coordinates[self.elements[el_num, (indices + 1) % 3]])
            
            # tack on additional nodes and note where to find them
            for k in range(3):
                if indices_to_lookup[where[k]] == 0:
                    indices_to_lookup[where[k]] = len(new_coordinates)
                    new_coordinates.append(midpoints[k, :])
                    
                    # store for output to other functions
                    have_encountered.append([entries[k]])
                    midpoints_mat[entries[k][0], 
                                  entries[k][1]] = len(new_coordinates) - 1
                
                midpoint_indices[k] = indices_to_lookup[where[k]]
                
                    
                
            # assemble the elements
            new_elements[4 * el_num, :] = np.array([
                self.elements[el_num, 0], 
                midpoint_indices[0], 
                midpoint_indices[-1]])
            new_elements[4 * el_num + 1, :] = np.array([
                midpoint_indices[0],
                self.elements[el_num, 1], 
                midpoint_indices[1]])
            new_elements[4 * el_num + 2, :] = np.array([
                midpoint_indices[-1],
                midpoint_indices[1],
                self.elements[el_num, 2]])
            new_elements[4 * el_num + 3, :] = np.array([
                midpoint_indices[0],
                midpoint_indices[1],
                midpoint_indices[-1]])
                
            
        # set everything
        self.set_coordinates(np.array(new_coordinates))
        self.set_elements(new_elements.astype(np.int))
         
        # this matrix must be symmetric
        midpoints_mat += midpoints_mat.T
        
        return midpoints_mat, have_encountered
            
    def pluck_new_element(self, element, new_indices):
        """
        Helper function for pink refine
        split an element into 4 others

        Parameters
        ----------
        element : 1 x 3 array
            element in question.
        new_indices : N x N sparse array
            indices of each midpoint.

        Returns
        -------
        new_elements : 4 x 3 array
            new elements.

        """
        new_elements = [
            [element[0], new_indices[element[0], element[1]], 
             new_indices[element[0], element[2]]], 
            [new_indices[element[0], element[1]], element[1], 
             new_indices[element[1], element[2]]], 
            [new_indices[element[0], element[2]], 
             new_indices[element[1], element[2]], element[2]],
            [new_indices[element[0], element[1]], 
             new_indices[element[1], element[2]], 
             new_indices[element[0], element[2]]]
        ]
        return new_elements
        
    def pink_refine(self):
        """
        A slower version of blue refine without loops, but which
        allows for higher degrees of mesh refinement

        Returns
        -------
        midpoints_indices : N x N sparse array
            table of midpoints indices.
        have_encountered : N long list
            list of edges encountered.

        """
        have_encountered = self.get_unique_edges()
        unique_edges = np.array(have_encountered)
        
        midpoints = 0.5 * (self.coordinates[unique_edges[:, 0], :] + 
                           self.coordinates[unique_edges[:, 1], :])
        
        new_coordinates = np.array(list(self.coordinates) + 
                                   list(midpoints))
        
        indices = np.arange(len(midpoints)) + self.num_nodes
        rows, cols = list(zip(*unique_edges))
        midpoints_indices = csr_matrix((indices, (rows, cols)), 
                                       shape=(self.num_nodes, self.num_nodes))
        
        midpoints_indices += midpoints_indices.T
                
        
        new_elements = list(map(
            lambda el: np.array(self.pluck_new_element(el, midpoints_indices)), 
            list(self.elements)))
        new_elements = np.concatenate(new_elements)
        
        
        self.num_nodes += len(have_encountered)
        self.num_elements *= 4
        self.elements = new_elements
        self.coordinates = new_coordinates
        
        
        return midpoints_indices, have_encountered
        
        
        
        
        
            
    def compute_barycenters(self):
        """
        Compute the barycenters for each element
        """

        # order coordinates for each triangle
        first_coords = self.coordinates[self.elements[:, 0], :]
        second_coords = self.coordinates[self.elements[:, 1], :]
        third_coords = self.coordinates[self.elements[:, 2], :]

        # barycenters as averages
        barycenters = (first_coords + second_coords + third_coords) / 3
        return barycenters
    
    def compute_affine_transformation_matrices(self):
        
        if self.B_matrices is not None:
            return self.B_matrices 
        
        else:
            # tensor where matrices[i, :, :] is a 2 x 2 matrix
            matrices = np.zeros([np.shape(self.elements)[0], 2, 2])
            
            # assign each entry for all matrices simultaneously
            for n in range(2):
                # column 1
                matrices[:, n, 0] = self.coordinates[self.elements[:, 1], n] - \
                    self.coordinates[self.elements[:, 0], n]
                    
                # column 2
                matrices[:, n, 1] = self.coordinates[self.elements[:, 2], n] - \
                    self.coordinates[self.elements[:, 0], n]
            self.B_matrices = matrices
                    
            return matrices           
        

    def affine_trans_det(self):
        """
        Compute determinant of affine transformation matrix
        """
        
        if self.det_B_vals is not None:
            return self.det_B_vals
        
        else:
            # order coordinates for each triangle
            first_coords = self.coordinates[self.elements[:, 0], :]
            second_coords = self.coordinates[self.elements[:, 1], :]
            third_coords = self.coordinates[self.elements[:, 2], :]
    
            # vectors forming sides of triangles
            xy12 = second_coords - first_coords
            xy13 = third_coords - first_coords
    
            # determinant via cross products
            det_B = xy12[:, 0] * xy13[:, 1] - xy13[:, 0] * xy12[:, 1]

        return det_B

    def affine_trans_mat_inverse(self):
        
        if self.B_inverses is not None:
            return self.B_inverses
        
        else:
            det_B = self.affine_trans_det()
            
            # start by scaling by determinants
            matrices = np.ones([np.shape(self.elements)[0], 2, 2]) / \
                np.tile(det_B, [2, 2, 1]).T
            
            # assign each element for each matrix in parallel
            # z_2y - z_0y
            matrices[:, 0, 0] *= self.coordinates[self.elements[:, 2], 1] - \
                self.coordinates[self.elements[:, 0], 1]
            # -(z_2x - z_0x)
            matrices[:, 0, 1] *= -(self.coordinates[self.elements[:, 2], 0] - 
                                  self.coordinates[self.elements[:, 0], 0])
            # -(z_1y - z_0y)
            matrices[:, 1, 0] *= -(self.coordinates[self.elements[:, 1], 1] - 
                                  self.coordinates[self.elements[:, 0], 1])
            # z_1x - z_0x
            matrices[:, 1, 1] *= self.coordinates[self.elements[:, 1], 0] - \
                self.coordinates[self.elements[:, 0], 0]
                
            return matrices

    #############################################################################
    """
    I am hesitant about using these functions defined here
    They may go better in another module
    """

    def gradient_lambda_functions(self):
        """
        compute gradients of lambda functions
        """

        # order coordinates for each triangle
        first_coords = self.coordinates[self.elements[:, 0], :]
        second_coords = self.coordinates[self.elements[:, 1], :]
        third_coords = self.coordinates[self.elements[:, 2], :]

        # vectors along each side of each element
        xy12 = second_coords - first_coords
        xy13 = third_coords - first_coords

        # affine transormation
        det_B = xy12[:, 0] * xy13[:, 1] - xy13[:, 0] * xy12[:, 1]

        lambda_derivatives = {}
        lambda_derivatives["dL1dx"] = (xy12[:, 1] - xy13[:, 1]) / det_B
        lambda_derivatives["dL1dy"] = xy13[:, 1] / det_B
        lambda_derivatives["dL2dx"] = (xy13[:, 0] - xy12[:, 0]) / det_B
        lambda_derivatives["dL2dy"] = -xy13 / det_B
        lambda_derivatives["dL3dx"] = -xy12[:, 1] / det_B
        lambda_derivatives["dL3dy"] = xy12[:, 0] / det_B

        return lambda_derivatives

    def gradient_operators(self):
        """
        compute gradient operator for mesh
        """

        lambda_derivatives = self.gradient_lambda_functions()

        gradient = {}
        gradient["ddx"] = [lambda_derivatives["dL1dx"],
                            lambda_derivatives["dL2dx"],
                            lambda_derivatives["dL3dx"]]
        gradient["ddy"] = [lambda_derivatives["dL1dy"],
                            lambda_derivatives["dL2dy"],
                            lambda_derivatives["dL3dy"]]

        return gradient

    def plot_mesh(self, show_indices=False, **kwargs):
        """
        Plot the mesh

        Parameters
        ----------
        show_indices : boolean, optional
            whether to show indices of each node on plot. The default is False.
        **kwargs : other arguments for plotting.

        Returns
        -------
        fig : matplotlib figure
            figure containing plot.
        ax : matplotlib figure axis
            axis containing plot.

        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        if not show_indices:
            symbol = "k^-"
        else:
            symbol = "k-"

        for element_no in range(self.num_elements):
            current_element = self.elements[element_no, :].astype(int)

            x_lst = list(self.coordinates[current_element, 0])
            y_lst = list(self.coordinates[current_element, 1])

            x_lst.append(x_lst[0])
            y_lst.append(y_lst[0])

            x_lst = np.array(x_lst)
            y_lst = np.array(y_lst)

            ax.plot(x_lst, y_lst, symbol, **kwargs)

        if show_indices:
            for coord_no in range(np.shape(self.coordinates)[0]):
                ax.annotate(str(coord_no), self.coordinates[coord_no, :],
                             fontsize=15, color="blue")
        return fig, ax

if __name__ == "__main__":
    
    import time
    
    coordinates = np.array([
        [0, 0],
        [1, 0],
        [0, 1]])
    elements = np.array([[0, 1, 2]])
    
    mesh = Mesh(coordinates=coordinates, elements=elements)
    
    num_refine = 8
    start = time.time()
    for j in range(num_refine):
        mesh.pink_refine()
    
    stop = time.time()
    print("time: ",stop - start)
    #mesh.plot_mesh()
    

