"""
Skeleton_tools
========

    Skeleton_tools is a python wrapper around the NetworkX (NX) Python package (https://networkx.lanl.gov/)
    to read, write and work with directed and undirected skeleton graphs.

"""

import networkx as nx
import numpy as np
from scipy.spatial import distance


class VP_type(object):
    def __init__(self, voxel=None, phys=None):
        self.voxel = voxel
        self.phys  = phys


class Skeleton(object):
    def __init__(self, identifier=None, voxel_size=None, seg_id=None, nx_graph=None):

        self.identifier = identifier
        self.voxel_size = voxel_size
        self.seg_id     = seg_id

        if nx_graph:
            assert isinstance(nx_graph, nx.Graph)
            self.nx_graph = nx_graph
        else:
            self.nx_graph = nx.Graph()


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


    def _add_edge_features(self, u, v, edge_feature_names=[]):
        '''
        internal function to add key:value to dictionary of one edge
        :param u:                   node_id of first node of edge
        :param v:                   node_id of second node of edge
        :param edge_feature_names:  key to be added
        :return:
        '''
        if 'length' in edge_feature_names:
            length = VP_type(voxel= np.sqrt(sum((self.nx_graph.node[u]['position'].voxel-self.nx_graph.node[v]['position'].voxel)**2)),
                             phys = np.sqrt(sum((self.nx_graph.node[u]['position'].phys - self.nx_graph.node[v]['position'].phys)**2)))
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


    def write_to_itk(self, outputfilename):
        '''

        :param outputfilename:
        :return:
        '''
        return


    def read_from_itk(self, inputfilename):
        '''

        :param inputfilename:
        :return:
        '''
        return


    def calcluate_total_path_length(self):
        '''

        :return:
        '''
        return


