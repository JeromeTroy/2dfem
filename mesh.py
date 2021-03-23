"""
Mesh class and related methods

Mesh class is the base for finite element (FE)
discretization.  This breaks up a domain into 
triangles.  Currently only designed for 2D domains

"""

import numpy as np
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
                
        return matrices           
        

    def affine_trans_det(self):
        """
        Compute determinant of affine transformation matrix
        """

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
        
        det_B = self.affine_trans_det()
        
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

    def plot_mesh(self, **kwargs):
        """
        Debugging method to plot mesh vertices and edges
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for element_no in range(self.num_elements):
            current_element = self.elements[element_no, :].astype(int)

            x_lst = list(self.coordinates[current_element, 0])
            y_lst = list(self.coordinates[current_element, 1])

            x_lst.append(x_lst[0])
            y_lst.append(y_lst[0])

            x_lst = np.array(x_lst)
            y_lst = np.array(y_lst)

            ax.plot(x_lst, y_lst, "k^-", **kwargs)

        return fig, ax


