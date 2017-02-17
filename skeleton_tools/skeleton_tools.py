"""
Skeleton_tools
========

    Skeleton_tools is a python wrapper around the NetworkX (NX) Python package (https://networkx.lanl.gov/)
    to read, write and work with directed and undirected skeleton graphs.

"""

import networkx as nx
import numpy as np
import os, copy
import knossos_utils
from scipy.spatial import KDTree
import operator


class VP_type(object):
    def __init__(self, voxel=None, phys=None):
        self.voxel = voxel
        self.phys = phys


class SkeletonContainer(object):
    """Collection of multiple skeleton objects."""

    def __init__(self, skeletons=None):
        if skeletons is not None:
            self.skeleton_list = skeletons
        else:
            self.skeleton_list = []

    def write_to_knossos_nml(self, outputfilename):
        # TODO make a generic function that takes as an argument a function and applies it to all the
        # skeletons in the list.
        for skeleton in self.skeleton_list:
            skeleton.fill_in_node_features('position_voxel')
        knossos_utils.from_nx_graphs_to_knossos([skeleton.nx_graph for
                                                 skeleton in self.skeleton_list],
                                                outputfilename)

    def read_from_knossos_nml(self, inputfilename, voxel_size=None):

        skeleton_list = knossos_utils.from_nml_to_nx_skeletons(inputfilename)
        if voxel_size is not None:
            for skeleton in skeleton_list:
                skeleton.voxel_size = voxel_size
        self.skeleton_list = skeleton_list

    def write_to_itk(self, outputfilename='data_test_itk', all_diameter_per_node=None, all_overwrite_existing=None):
        """ Write all skeleton in container to itk format (see example file in skeleton_tools/test_data_itk.txt)

        Parameters
        ----------
            outputfilename: string
                path and name where to store file to, for ever skeleton its self.identifier is added, s.t. outputfilename_identifier.txt
            all_diameter_per_node: List of numpy array, shape: [num_skeletons x [1 x number_of_nodes())], default: 2.*np.ones(self.nx_graph.number_of_nodes()) for every skeleton
                array to define the diameter for every node. Represents the diameter shown in the viewer not the
                actual diameter of the neuron.
            overwrite_existing: list of bool, shape: [num_skeletons]
                if True, overwrite existing file with the same name, if False: create new file or assertion error

        """

        for nr_skeleton, skeleton in enumerate(self.skeleton_list):
            if all_diameter_per_node is None:
                diameter_per_node = None
            else:
                diameter_per_node = all_diameter_per_node[nr_skeleton]

            if all_overwrite_existing is None:
                overwrite_existing = False
            else:
                overwrite_existing = all_overwrite_existing[nr_skeleton]

            skeleton.write_to_itk(outputfilename=outputfilename + '_' + str(skeleton.identifier),
                                  diameter_per_node=diameter_per_node, overwrite_existing=overwrite_existing)

    def from_skeletons_to_binary_mask(self, mask_shape, thickness=4):
        """ Writes all skeletons into a single volume (as a binary mask).
        Parameters
        ----------
        mask_shape: shape of the newly created volume.
        thickness: the length of the cube each node is represented with in the volume. Set to 1 if only the voxel
        position itself should be marked.
        """
        thickness //= 2
        mask = np.zeros(mask_shape, dtype=np.uint8)
        for skeleton in self.skeleton_list:
            for _, node_dic in skeleton.nx_graph.nodes_iter(data=True):
                voxel_pos = node_dic['position'].voxel
                x, y, z = voxel_pos
                mask[x-thickness:x+thickness, y-thickness:y+thickness, z-thickness:z+thickness] = 1
        return mask



class Skeleton(object):
    def __init__(self, identifier=None, voxel_size=None, seg_id=None, nx_graph=None):

        self.identifier = identifier
        if isinstance(voxel_size, list):
            voxel_size = np.array(voxel_size)
        if voxel_size is not None:
            assert voxel_size.shape[0] == 3
        self.voxel_size = voxel_size
        self.seg_id = seg_id

        if nx_graph:
            assert isinstance(nx_graph, nx.Graph)
            self.nx_graph = nx_graph
        else:
            self.nx_graph = nx.Graph()

    def initialize_from_datapoints(self, datapoints, vp_type_voxel, edgelist=None, datapoints_type='nparray'):
        """ Initializes a graph with provided datapoints. The node_id in the graph corresponds to the index number of the array.

        Parameters
        ----------
            datapoints: numpy array, shape: N x 3
                Array with point coordinates
            vp_type_voxel: bool
                Whether the provided coordinates have voxel dimension or correspond to the physical coordinates.
            edgelist: list with 2d tuples or 2d lists
                If set, edges are added to the nxgraph.

        """
        assert self.nx_graph.number_of_nodes() == 0, 'Graph is not empty, can not initialize a graph.'

        if datapoints_type == 'nparray':
            assert isinstance(datapoints,
                              np.ndarray), 'datapoints is not of type numpy array, either change input type or set datapoints_type differently'
        if datapoints_type == 'dic':
            assert isinstance(datapoints,
                              dict), 'datapoints is not of type dictionary, either change input type or set datapoints_type differently'

        if datapoints_type == 'nparray':
            assert datapoints.shape[1] == 3
            if edgelist is not None:
                assert datapoints.shape[0] == len(np.unique(
                    np.array(edgelist))), 'number of nodes (extracted from edges) and provided coordinates do not match'
            # TODO: Add a function that allows to add many coordinates at once to an existing graph, but tricky
            # since how to handle the Node IDs.

            for node_id in range(datapoints.shape[0]):
                pos = datapoints[node_id, :]
                if vp_type_voxel:
                    self.add_node(node_id, pos_voxel=pos)
                else:
                    self.add_node(node_id, pos_phys=pos)
        elif datapoints_type == 'dic':
            for node_id, pos in datapoints.iteritems():
                if vp_type_voxel:
                    self.add_node(node_id, pos_voxel=pos)
                else:
                    self.add_node(node_id, pos_phys=pos)
        else:
            print 'provided datapoints_type not supported'


        if edgelist is not None:
            self.nx_graph.add_edges_from(edgelist)


    def initialize_from_l1_graph(self, l1_graph):
        assert self.nx_graph.number_of_nodes() == 0
        
        for node_id in xrange(l1_graph.num_nodes):
            self.add_node(node_id, pos_voxel=l1_graph.positions[node_id])
            self.nx_graph.node[node_id]['orientation'] = l1_graph.orientations[node_id]
            self.nx_graph.node[node_id]['partner'] = l1_graph.partner[node_id] 
       
        self.nx_graph.add_edges_from(l1_graph.edges)

        

    def add_node(self, node_id, pos_voxel=None, pos_phys=None):
        """ Add node to exisiting skeleton

        Parameters
        ----------
            node_id:    node_id of node to be added
            pos_voxel:  np.array, int
                    Position of node in voxel-coordinates
            pos_phys:   np.array, float
                    Position of node in physical-coordinates
        """

        position = VP_type(voxel=pos_voxel, phys=pos_phys)
        self.nx_graph.add_node(n=node_id, position=position)


    def add_edge(self, u, v):
        """ Add edge to existing skeleton

        Parameters
        ----------
            u:  node_id of first node of edge
            v:  node_id of second node of edge

        """

        self.nx_graph.add_edge(u, v)


    def _add_node_features(self, node_id, node_feature_names=[]):
        """ Internal function to add key:value to dictionary of one node

        Parameters
        ----------
            node_id:            node_id of node where feature is added
            node_feature_names: key to be added
        """

        if 'direction' in node_feature_names:
            pass

        if 'position_voxel' in node_feature_names:
            # Complement the voxel
            if self.nx_graph.node[node_id]['position'].voxel is None:
                assert self.voxel_size is not None, 'can not convert positions into voxel space, since voxel size is not given'
                self.nx_graph.node[node_id]['position'].voxel = (self.nx_graph.node[node_id][
                                                                     'position'].phys // self.voxel_size).astype(np.int)

        if 'position_phys' in node_feature_names:
            if self.nx_graph.node[node_id]['position'].phys is None:
                assert self.voxel_size is not None, 'can not convert positions into phys space, since voxel size is not given'
                self.nx_graph.node[node_id]['position'].phys = self.nx_graph.node[node_id]['position'].voxel * self.voxel_size


    def _add_edge_features(self, u, v, edge_feature_names=[]):
        """ Internal function to add key:value to dictionary of one edge

        Parameters
        ----------
            u:                   node_id of first node of edge
            v:                   node_id of second node of edge
            edge_feature_names:  key to be added

        """

        if 'length' in edge_feature_names:
            pos_u = self.nx_graph.node[u]['position']
            pos_v = self.nx_graph.node[v]['position']
            length = VP_type()

            if pos_u.voxel is not None and pos_v.voxel is not None:
                length.voxel = np.sqrt(sum((pos_u.voxel - pos_v.voxel) ** 2))

            if pos_u.phys is not None and pos_v.phys is not None:
                length.phys = np.sqrt(sum((pos_u.phys - pos_v.phys) ** 2))

            self.nx_graph.edge[u][v]['length'] = length

        if 'direction' in edge_feature_names:
            pass


    def fill_in_node_features(self, node_feature_names=[]):
        """ Add key:value to all dictionaries of all nodes

        Parameters
        ----------
            node_feature_names: key to be added

        """
        for node_id in self.nx_graph.nodes_iter():
            self._add_node_features(node_id=node_id, node_feature_names=node_feature_names)


    def fill_in_edge_features(self, edge_feature_names=[]):
        """ Add key:value to all dicitonaries of all edges

         Parameters
         ----------
             edge_feature_names: key to be added
         """
        for u, v in self.nx_graph.edges_iter():
            self._add_edge_features(u, v, edge_feature_names=edge_feature_names)


    def write_to_knossos_nml(self, outputfilename):
        return


    def read_from_knossos_nml(self, inputfilename):
        return


    def write_to_itk(self, outputfilename='data_test_itk', skeleton_id=None, diameter_per_node=None, overwrite_existing=False):
        """ Write skeleton to itk format (see example file in skeleton_tools/test_data_itk.txt). The physical coordinates
            are written to the itk format.

        Parameters
        ----------
            outputfilename: string
                path and name where to store file to
            skeleton_id:        int,
                Colour ID of skeleton in Volume Viewer (e.g. 0: black)
            diameter_per_node:  numpy array, shape: [1 x number_of_nodes()), default: 2.*np.ones(self.nx_graph.number_of_nodes())
                array to define the diameter for every node. Represents the diameter shown in the viewer not the
                actual diameter of the neuron.
            overwrite_existing: bool
                if True, overwrite existing file with the same name, if False: create new file or assertion error

        Note
        ----
            When looking at the images with the volume viewer, please add --resX=self.voxel_size[2] --resY=self.voxel_size[1]
                --resZ=self.voxel_size[0] to appropriately scale the skeleton


        """
        assert self.nx_graph is not None

        if not overwrite_existing:
            assert not os.path.exists(outputfilename + '.txt'), "outputfilename exists already: " + str(
                outputfilename) + '.txt'

        with open(outputfilename + '.txt', 'w') as d_file:
            np.savetxt(d_file, np.array(["ID " + str(skeleton_id)]), '%s')
            np.savetxt(d_file, np.array(["POINTS " + str(self.nx_graph.number_of_nodes()) + " FLOAT"]), '%s')
            for node_id, node_attr in self.nx_graph.nodes_iter(data=True):
                if node_attr['position'].phys is None:
                    assert self.voxel_size is not None
                    assert self.nx_graph.node[node_id]['position'].voxel is not None
                    position = node_attr['position'].voxel * self.voxel_size
                else:
                    position = node_attr['position'].phys
                np.savetxt(d_file, np.fliplr(np.reshape(position, (1, 3))), '%i')

            np.savetxt(d_file, np.array(["\nEDGES " + str(self.nx_graph.number_of_edges())]), '%s')
            for u, v in self.nx_graph.edges_iter():
                np.savetxt(d_file, np.array([[u, v]]), '%i')

            np.savetxt(d_file, np.array([" \ndiameters 0 0 FLOAT"]), '%s')
            if diameter_per_node is None:
                diameter_per_node = 2. * np.ones(self.nx_graph.number_of_nodes())
            else:
                assert len(diameter_per_node) == self.nx_graph.number_of_nodes()
            np.savetxt(d_file, diameter_per_node, '%.4e')
        return


    def read_from_itk(self, inputfilename):
        return


    def calculate_total_phys_length(self):
        """ Calculate total phys length. """
        total_path_length = 0
        self.fill_in_node_features('position_phys')
        for u, v, edge_attr in self.nx_graph.edges_iter(data=True):
            if not 'length' in edge_attr:
                self._add_edge_features(u, v, 'length')
            path_length = edge_attr['length'].phys
            total_path_length += path_length
        return total_path_length


    def _interpolate_edge(self, u, v, step_size, VP_type):
        # VP_type: 'voxel' or 'phys'

        if not 'length' in self.nx_graph.edge[u][v]:
                self._add_edge_features(u, v, 'length')
        if VP_type == 'voxel':
            length_step = self.nx_graph.edge[u][v]['length'].voxel
            assert length_step is not None
            if length_step <= step_size:
                return
            pos_u = self.nx_graph.node[u]['position'].voxel
            pos_v = self.nx_graph.node[v]['position'].voxel

        elif VP_type == 'phys':
            length_step = self.nx_graph.edge[u][v]['length'].phys
            assert length_step is not None
            if length_step <= step_size:
                return
            pos_u = self.nx_graph.node[u]['position'].phys
            pos_v = self.nx_graph.node[v]['position'].phys

        dir_vec = pos_v - pos_u
        dir_vec = dir_vec/np.linalg.norm(dir_vec)

        number_of_new_nodes = int(np.ceil(length_step/step_size)-1)
        ori_pos  = pos_u
        cur_node = u
        cur_pos  = pos_u.copy()
        min_one_node_added = False

        max_node_id = np.max(self.nx_graph.nodes())
        for step in range(number_of_new_nodes):

            if VP_type == 'voxel':
                new_pos = (ori_pos + step*dir_vec*step_size).astype(np.int)
            elif VP_type == 'phys':
                new_pos = (ori_pos + step * dir_vec * step_size).astype('float')

            if np.equal(cur_pos, new_pos).all():
                # This happens if the direction_vector steps is too small to have an effect in a single step.
                continue

            new_node_id = max_node_id.copy()+1
            max_node_id = new_node_id.copy()
            if VP_type == 'voxel':
                self.add_node(new_node_id, pos_voxel=new_pos)
                self.nx_graph.add_edge(cur_node, new_node_id)
            elif VP_type == 'phys':
                self.add_node(new_node_id, pos_phys=new_pos)
                self.nx_graph.add_edge(cur_node, new_node_id)
            min_one_node_added = True
            cur_node = new_node_id.copy()
            cur_pos = new_pos.copy()

        # Add final edge to he last inserted new node and the original v node and remove original edge u, v.
        if min_one_node_added:
            self.nx_graph.add_edge(cur_node, v)
            self.nx_graph.remove_edge(u, v)

    def _interpolate_edge_linebased(self, u, v, step_size, VP_type):
        if not 'length' in self.nx_graph.edge[u][v]:
            self._add_edge_features(u, v, 'length')

        if VP_type == 'voxel':
            length_step = self.nx_graph.edge[u][v]['length'].voxel
            assert length_step is not None
            if length_step <= step_size:
                return
            pos_u = self.nx_graph.node[u]['position'].voxel
            pos_v = self.nx_graph.node[v]['position'].voxel

        elif VP_type == 'phys':
            length_step = self.nx_graph.edge[u][v]['length'].phys
            assert length_step is not None
            if length_step <= step_size:
                return
            pos_u = self.nx_graph.node[u]['position'].phys
            pos_v = self.nx_graph.node[v]['position'].phys

        dda = DDA3(pos_u, pos_v)
        coord_on_line = dda.draw()
        coord_on_line = coord_on_line[1:-1] # Start and end are included in the list.
        cur_node = u

        for ii, coord in enumerate(coord_on_line):
            new_node_id = np.max(self.nx_graph.nodes())+1
            if VP_type == 'voxel':
                self.add_node(new_node_id, pos_voxel=coord)
                self.nx_graph.add_edge(cur_node, new_node_id)
            cur_node = new_node_id.copy()

        # Add final edge to he last inserted new node and the original v node and remove original edge u, v.
        if len(coord_on_line) > 0:
            self.nx_graph.add_edge(cur_node, v)
            self.nx_graph.remove_edge(u, v)


    def interpolate_edges(self, step_size=1, VP_type=None):
        """ Interpolates all edges, meaning inserting additional nodes such that the length from one node to the
        other corresponds to voxel_step_size.

        Parameters
        ----------
            voxel_step_size: int
                How long the distance between to consecutive nodes should be.
        Notes
        ---------
            Note that the node IDs of the orginal graph stay the same. The additional inserted nodes get new IDs
            starting with the ID number of the current max node id. Consecutive node IDs thus not necessarily
            correspond to consecutive nodes in space.

        """
        if VP_type == 'voxel':
            self.fill_in_node_features('position_voxel')
        elif VP_type == 'phys':
            self.fill_in_node_features('position_phys')
        if step_size > 1 or VP_type == 'phys':
            print 'using old interpolation function, new (better) interpolation function only implemented for ' \
                  'voxel_size = 1 and VP_type = voxel, parameter set to ' \
                  'voxel_size = %i and VP_type = %s' %(step_size, VP_type)

        # Get the list of all edges, before additional edges are inserted
        edges = self.nx_graph.edges()
        for u, v in edges:
            if step_size > 1 or VP_type == 'phys':
                self._interpolate_edge(u, v, step_size=step_size, VP_type=VP_type)
            else:
                self._interpolate_edge_linebased(u, v, step_size=step_size, VP_type=VP_type)

        # Remove edge dictionary, since features could have changed (such as direction vector).
        for edge_id in self.nx_graph.edges_iter():
            self.nx_graph.edge[edge_id] = {}



    def get_node_ids_of_endpoints(self):
        """ Return all node_ids which are endpoints (= have only one neighboring node). """
        all_node_ids_of_endpoints = []
        for node_id in self.nx_graph.nodes_iter(data=False):
            if len(self.nx_graph.neighbors(node_id)) == 1:
                all_node_ids_of_endpoints.append(node_id)
        return all_node_ids_of_endpoints


    def from_nx_skeleton_to_datapoints(self, VP_type):
        """ Create array of node positions
         Parameters
        ----------
            VP_type: string, 'voxel' or 'phys'
                if voxel or phys position should be written to array
         Returns
        ----------
            dpts: np.array [N x 3]
                Numpy array with node positions either voxel or phys coordinates
        Notes
        ---------
            Order of node positions is arbitrary, not specified by edges
        """
        node_size = len(self.nx_graph)
        dpts = np.empty((node_size, 3))
        for index_count, (node_id, node_att) in enumerate(self.nx_graph.nodes_iter(data=True)):
            if VP_type == 'voxel':
                node_pos = node_att['position'].voxel
            elif VP_type == 'phys':
                node_pos = node_att['position'].phys
            dpts[index_count, :] = node_pos
        return dpts


    def get_kdtree_from_datapoints(self, VP_type):
        """ Add kdtree_of_nodes as attribute to SKeleton instance. kdtree_of_nodes is a KD tree
         Parameters
        ----------
            VP_type: string, 'voxel' or 'phys'
                if voxel or phys position should be written to array
         Returns
        ----------
            tree: KD tree
                Instance of scipy.spatial KD tree class, containing voxel or phys node positions
        """

        data = self.from_nx_skeleton_to_datapoints(VP_type=VP_type)
        tree = KDTree(zip(data[:, 0].ravel(), data[:, 1].ravel(), data[:, 2].ravel()))
        self.kdtree_of_nodes = tree
        return tree


    def get_precision(self, other, tolerance_distance):
        """ get precision of predicted skeleton (other) compared to target skeleton (self).
            A node is counted as correct if lies within tolerance distance to any node along the
            edge-interpolated target skeleton.'
         Parameters
        ----------
            other: Skeleton(),
                Instance of Skeleton class containing the predicted nodes
            tolerance_distance: int,
                distance within a nodes must lie to closest node in target skeleton to be counted as correct
         Returns
        ----------
            precision: float,
                precision: true_positives / (true_positives+false_positives)
                           percentage of nodes in other which lie within tolerance_distance to any node in self
         Notes
        ---------
            Only the self/target skeleton is edge-interpolated
        """
        copy_self = copy.deepcopy(self)
        assert self.nx_graph.number_of_nodes() == copy_self.nx_graph.number_of_nodes()

        copy_self.interpolate_edges(step_size=1, VP_type='voxel')

        # create kdtree if does not exist yet
        if not hasattr(copy_self, 'kdtree_of_nodes'):
            copy_self.get_kdtree_from_datapoints(VP_type='voxel')
        if not hasattr(other, 'kdtree_of_nodes'):
            other.get_kdtree_from_datapoints(VP_type='voxel')

        # iterate over pred nodes and check in target_skeleton for nodes closer than tolerance_distance
        # get list which contains one list of close_nodes for every node in other/predicted skeleton
        close_nodes_per_pred_node = other.kdtree_of_nodes.query_ball_tree(other=copy_self.kdtree_of_nodes, r=tolerance_distance)
        total_nodes_pred = other.nx_graph.number_of_nodes()
        only_nonempty_close_nodes = filter(None, close_nodes_per_pred_node)
        num_correct_nodes = float(len(only_nonempty_close_nodes))

        precision = num_correct_nodes / total_nodes_pred
        del copy_self
        return precision


    def get_recall(self, other, tolerance_distance):
        """ get recall of target skeleton (self) compared to predicted skeleton (other).
            A node is counted as recalled if any node in the edge-interpolated other/predicted skeleton lies within
            tolerance distance to this specific node in self/target skeleton.'
         Parameters
        ----------
            other: Skeleton(),
                Instance of Skeleton class containing the predicted nodes
            tolerance_distance: int,
                distance within which a predicted nodes must lie for a target node to be counted as recalled
         Returns
        ----------
            recall: float,
                recall:  true_positives / (true_positives+false_positives)
                         percentage of nodes in self which have at least on node in other which lies within
                         tolerance_distance to it
        Notes
        ---------
            Only the predicted skeleton is edge-interpolated
        """
        copy_other = copy.deepcopy(other)
        assert other.nx_graph.number_of_nodes() == copy_other.nx_graph.number_of_nodes()

        # other = predicted skeleton, self = target_skeleton
        if not hasattr(self, 'kdtree_of_nodes'):
            self.get_kdtree_from_datapoints(VP_type='voxel')
        copy_other.interpolate_edges(step_size=1, VP_type='voxel')
        if not hasattr(copy_other, 'kdtree_of_nodes'):
            copy_other.get_kdtree_from_datapoints(VP_type='voxel')

        # iterate over target nodes and check in predicted nodes for nodes closer than tolerance_distance indicating
        # that this target node was recalled
        # get list which contains one list of close_nodes for every node in self/target_skeleton
        close_nodes_per_target_node = self.kdtree_of_nodes.query_ball_tree(other=copy_other.kdtree_of_nodes, r=tolerance_distance)
        total_num_nodes = self.nx_graph.number_of_nodes()
        only_nonempty_close_nodes = filter(None, close_nodes_per_target_node)
        num_recalled_nodes = float(len(only_nonempty_close_nodes))

        recall = num_recalled_nodes / total_num_nodes
        del copy_other
        return recall


    def get_distance_to_skeleton(self, x):
        """Get for every node in other the closest distance to the interpolated skeleton in self.nx_graph
         Parameters
        ----------
            x:    array, [N x 3]
                Datapoints to get clostest distance to interpolated self.nx_graph
         Returns
         ---------
            distance_to_cl: array of floats, [N,]
                min. distance to centerline for every point in distant_pts

        """
        copy_self = copy.deepcopy(self)
        assert self.nx_graph.number_of_nodes() == copy_self.nx_graph.number_of_nodes()

        copy_self.interpolate_edges(step_size=1, VP_type='voxel')
        if not hasattr(copy_self, 'kdtree_of_nodes'):
            copy_self.get_kdtree_from_datapoints(VP_type='voxel')
        # "query()" returns distance to closest points AND their location, here only distance considered
        distance_to_cl = copy_self.kdtree_of_nodes.query(x=x, k=1, eps=0, p=2, distance_upper_bound=np.inf)[0]
        del copy_self
        return distance_to_cl

    def apply_transformation(self, transformation):
        """Applies a transformation matrix to coordinates of the graph e.g. skeleton is rotated based on
        transformation matrix.
         Parameters
        ----------
            transformation:    array, [3 x Z x Y x X]
                Map with new coordination, e.g. coord [0, 0, 0] is mapped to transformation[:, 0, 0, 0]

        Notes
        ---------
            Nodes and edges IDs stay the same, but skeleton is altered in space. Skeleton might contain negative
            coordinates.
        """
        for node_id, node_dict in self.nx_graph.nodes_iter(data=True):
            cur_pos_voxel = node_dict['position'].voxel.astype(np.int)
            x, y, z = cur_pos_voxel
            z, y, x = transformation[:, z, y, x]
            new_pos_voxel = np.array([x, y, z])
            new_pos_voxel = new_pos_voxel.astype(np.int)

            if self.voxel_size is not None:
                new_pos_phys = new_pos_voxel*self.voxel_size
            else:
                new_pos_phys = None
            self.add_node(node_id, pos_voxel=new_pos_voxel, pos_phys=new_pos_phys)

        # Remove edge dictionary, since features could have changed (such as direction vector).
        for edge_id in self.nx_graph.edges_iter():
            self.nx_graph.edge[edge_id] = {}


def dda_round(x):
    return (x + 0.5).astype(int)


class DDA3:
    def __init__(self, start, end, scaling=np.array([1, 1, 1])):
        assert (start.dtype == int)
        assert (end.dtype == int)

        self.start = (start * scaling).astype(float)
        self.end = (end * scaling).astype(float)
        self.line = [dda_round(self.start)]

        self.max_direction, self.max_length = max(enumerate(abs(self.end - self.start)), key=operator.itemgetter(1))
        self.dv = (self.end - self.start) / self.max_length

    def draw(self):
        for step in range(int(self.max_length)):
            self.line.append(dda_round((step + 1) * self.dv + self.start))

        assert (np.all(self.line[-1] == self.end))

        for n in xrange(len(self.line) - 1):
            assert (np.linalg.norm(self.line[n + 1] - self.line[n]) <= np.sqrt(3))

        return self.line






