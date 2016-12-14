"""
Skeleton_tools
========

    Skeleton_tools is a python wrapper around the NetworkX (NX) Python package (https://networkx.lanl.gov/)
    to read, write and work with directed and undirected skeleton graphs.

"""

import networkx as nx
import numpy as np
import os
# import knossos_utils


class VP_type(object):
    def __init__(self, voxel=None, phys=None):
        self.voxel = voxel
        self.phys = phys


class SkeletonContainer(object):
    """Collection of multiple skeleton objects."""

    def __init__(self, skeletons):
        self.skeleton_list = skeletons

    def write_to_knossos_nml(self, outputfilename):
        # TODO make a generic function that takes as an argument a function and applies it to all the
        # skeletons in the list.
        for skeleton in self.skeleton_list:
            skeleton.fill_in_node_features('position_voxel')
        knossos_utils.from_nx_graphs_to_knossos([skeleton.nx_graph for
                                                 skeleton in self.skeleton_list],
                                                outputfilename)



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

            skeleton.write_to_itk(outputfilename=outputfilename+'_'+str(skeleton.identifier), diameter_per_node=diameter_per_node, overwrite_existing=overwrite_existing)



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

    def initialize_from_datapoints(self, datapoints, vp_type_voxel, edgelist=None):
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
        assert datapoints.shape[1] == 3
        assert datapoints.shape[0] == len(np.unique(
            np.array(edgelist))), 'number of nodes (extracted from edges) and provided coordinates do not match'
        # TODO: Add a function that allows to add many coordinates at once to an exisiting graph, but tricky
        # since how to handle the Node IDs.
        assert self.nx_graph.number_of_nodes() == 0, 'Graph is not empty, can not initialize a graph.'
        for node_id in range(datapoints.shape[0]):
            pos = datapoints[node_id, :]
            if vp_type_voxel:
                self.add_node(node_id, pos_voxel=pos)
            else:
                self.add_node(node_id, pos_phys=pos)

        if edgelist is not None:
            self.nx_graph.add_edges_from(edgelist)

    def add_node(self, node_id, pos_voxel=None, pos_phys=None):
        '''
        add node to exisiting skeleton
        :param node_id:     node_id of node to be added
        :param pos_voxel:   position of node in voxel-coordinates (int)
        :param pos_phys:    position of node in phsyical coordinates (float)
        :return:            -
        '''
        position = VP_type(voxel=pos_voxel, phys=pos_phys)
        self.nx_graph.add_node(n=node_id, position=position)

    def add_edge(self, u, v):
        '''
        add edge to existing skeleton
        :param u:   node_id of first node of edge
        :param v:   node_id of second node of edge
        :return:    -
        '''
        self.nx_graph.add_edge(u, v)

    def _add_node_features(self, node_id, node_feature_names=[]):
        '''
        internal function to add key:value to dictionary of one node.
        :param node_id:             node_id of node where feature is added
        :param node_feature_names:  key to be added
        :return:
        '''
        if 'direction' in node_feature_names:
            pass

        if 'position_voxel' in node_feature_names:
            # Complement the voxel
            assert self.voxel_size is not None, 'can not convert positions into voxel space, since voxel size is not given'
            self.nx_graph.node[node_id]['position'].voxel = (self.nx_graph.node[node_id][
                                                                'position'].phys // self.voxel_size).astype(np.int)

        if 'position_phys' in node_feature_names:
            # Complement the voxel
            assert self.voxel_size is not None, 'can not convert positions into phys space, since voxel size is not given'
            self.nx_graph.node[node_id]['position'].phys = self.nx_graph.node[node_id][
                                                               'position'].voxel * self.voxel_size

    def _add_edge_features(self, u, v, edge_feature_names=[]):
        '''
        internal function to add key:value to dictionary of one edge
        :param u:                   node_id of first node of edge
        :param v:                   node_id of second node of edge
        :param edge_feature_names:  key to be added
        :return:
        '''
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
        '''
        add key:value to all dictionaries of all nodes
        :param node_feature_names:      key to be added
        :return:
        '''
        for node_id in self.nx_graph.nodes_iter():
            self._add_node_features(node_id=node_id, node_feature_names=node_feature_names)

    def fill_in_edge_features(self, edge_feature_names=[]):
        '''
        add key:value to all dicitonaries of all edges
        :param edge_feature_names:      key to be added
        :return:
        '''

        for u, v in self.nx_graph.edges_iter():
            self._add_edge_features(u, v, edge_feature_names=edge_feature_names)

    def write_to_knossos_nml(self, outputfilename):
        '''

        :param outputfilename:
        :return:
        '''
        return

    def read_from_knossos_nml(self, inputfilename):
        '''

        :param inputfilename:
        :return:
        '''
        return

    def write_to_itk(self, outputfilename='data_test_itk', diameter_per_node=None, overwrite_existing=False):
        """ Write skeleton to itk format (see example file in skeleton_tools/test_data_itk.txt)

        Parameters
        ----------
            outputfilename: string
                path and name where to store file to
            diameter_per_node: numpy array, shape: [1 x number_of_nodes()), default: 2.*np.ones(self.nx_graph.number_of_nodes())
                array to define the diameter for every node. Represents the diameter shown in the viewer not the
                actual diameter of the neuron.
            overwrite_existing: bool
                if True, overwrite existing file with the same name, if False: create new file or assertion error

        """

        assert self.nx_graph is not None

        if not overwrite_existing:
            assert not os.path.exists(outputfilename+'.txt'), "outputfilename exists already: "+str(outputfilename)+'.txt'


        with open(outputfilename+'.txt', 'w') as d_file:
            np.savetxt(d_file, np.array(["ID " + str(self.seg_id)]), '%s')
            np.savetxt(d_file, np.array(["POINTS " + str(self.nx_graph.number_of_nodes()) + " FLOAT"]), '%s')
            for node_id, node_attr in self.nx_graph.nodes_iter(data=True):
                if node_attr['position'].voxel is None:
                    assert self.voxel_size is not None
                    assert self.nx_graph.node[node_id]['position'].phys is not None
                    position = node_attr['position'].phys * self.voxel_size
                else:
                    position = node_attr['position'].voxel
                np.savetxt(d_file, np.fliplr(np.reshape(position, (1, 3))), '%i')

            np.savetxt(d_file, np.array(["\nEDGES " + str(self.nx_graph.number_of_edges())]), '%s')
            for u, v in self.nx_graph.edges_iter():
                np.savetxt(d_file, np.array([[u, v]]), '%i')

            np.savetxt(d_file, np.array([" \ndiameters 0 0 FLOAT"]), '%s')
            if diameter_per_node is None:
                diameter_per_node = 2.*np.ones(self.nx_graph.number_of_nodes())
            else:
                assert len(diameter_per_node) == self.nx_graph.number_of_nodes()
            np.savetxt(d_file, diameter_per_node, '%.4e')

        return

    def read_from_itk(self, inputfilename):
        '''

        :param inputfilename:
        :return:
        '''


        return

    def calculate_total_phys_length(self):
        """Calculate total phys length."""
        total_path_length = 0
        for u, v, edge_attr in self.nx_graph.edges_iter(data=True):
            if not 'length' in edge_attr:
                self._add_edge_features(u, v, 'length')
            path_length = edge_attr['length'].phys
            total_path_length += path_length
        return total_path_length
