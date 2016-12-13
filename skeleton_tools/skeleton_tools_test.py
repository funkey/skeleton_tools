import numpy as np
import unittest
from skeleton_tools import Skeleton, VP_type
import networkx as nx
from scipy.spatial import distance



class TestSkeletonTools(unittest.TestCase):


    def test_basicinitialization(self):
        my_identifier = 3
        my_voxel_size = np.array([1., 1., 2.])
        my_seg_id = 2
        my_nx_graph = nx.Graph()

        my_skeleton = Skeleton(identifier=my_identifier, voxel_size=my_voxel_size, seg_id=my_seg_id)
        my_skeleton_with_graph = Skeleton(identifier=my_identifier, voxel_size=my_voxel_size, seg_id=my_seg_id, nx_graph=my_nx_graph)

        self.assertEqual(my_skeleton.nx_graph.number_of_nodes(), 0)
        self.assertEqual(my_skeleton.nx_graph.number_of_edges(), 0)
        self.assertEqual(my_skeleton.identifier, my_identifier)
        self.assertTrue(np.array_equal(my_skeleton.voxel_size, my_voxel_size))
        self.assertEqual(my_skeleton.seg_id, my_seg_id)
        self.assertIsInstance(my_skeleton.nx_graph, nx.Graph)
        self.assertIsInstance(my_skeleton_with_graph.nx_graph, nx.Graph)


    def test_addnode(self):
        my_skeleton = Skeleton()
        scaling = np.array([1., 1., 0.5])
        nodes_pos_phys = np.array([[0, 0, 0], [50, 50, 50], [100, 100, 100], [150, 150, 150]])
        nodes_pos_voxel = np.floor(nodes_pos_phys*scaling)
        node_ids = range(nodes_pos_voxel.shape[0])
        for node_id, pos_voxel, pos_phys in zip(node_ids, nodes_pos_voxel, nodes_pos_phys):
            my_skeleton.add_node(node_id, pos_voxel=pos_voxel, pos_phys=pos_phys)
        # self.assertEqual(my_skeleton.nx_graph.number_of_nodes(), nodes_pos_voxel.shape[0])
        # self.assertEqual(my_skeleton.nx_graph.node[2]['position'].voxel, nodes_pos_voxel[2])
        # self.assertEqual(my_skeleton.nx_graph.node[2]['position'].phys, nodes_pos_phys[2])


    def test_addedge(self):
        my_skeleton = Skeleton()
        edges = [(0, 1), (1, 2), (2, 3)]
        for edge in edges:
            my_skeleton.add_edge(u=edge[0], v=edge[1])
        self.assertEqual(my_skeleton.nx_graph.number_of_nodes(), len(edges)+1)
        self.assertEqual(my_skeleton.nx_graph.number_of_edges(), len(edges))


    def test_addedgefeatures(self):
        my_skeleton = Skeleton()
        scaling = np.array([1., 1., 0.5])
        nodes_pos_phys = np.array([[0, 0, 0], [50, 50, 50], [100, 100, 100], [150, 150, 150]])
        nodes_pos_voxel = np.floor(nodes_pos_phys*scaling)
        node_ids = range(nodes_pos_voxel.shape[0])
        for node_id, pos_voxel, pos_phys in zip(node_ids, nodes_pos_voxel, nodes_pos_phys):
            my_skeleton.add_node(node_id=node_id, pos_voxel=pos_voxel, pos_phys=pos_phys)

        edges = [(0, 1), (1, 2), (2, 3)]
        for edge in edges:
            my_skeleton.add_edge(u=edge[0], v=edge[1])

        my_edge = 2
        edge_u = edges[my_edge][0]
        edge_v = edges[my_edge][1]
        my_skeleton._add_edge_features(u=edge_u, v=edge_v, edge_feature_names=['length'])

        length_voxel = np.sqrt(sum((nodes_pos_voxel[edge_u] - nodes_pos_voxel[edge_v])**2))
        length_phys  = np.sqrt(sum((nodes_pos_phys[edge_u] - nodes_pos_phys[edge_v])**2))
        self.assertEqual(my_skeleton.nx_graph.edge[edge_u][edge_v]['length'].voxel, length_voxel)
        self.assertEqual(my_skeleton.nx_graph.edge[edge_u][edge_v]['length'].phys, length_phys)

