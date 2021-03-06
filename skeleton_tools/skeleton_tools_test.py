import numpy as np
import unittest
from skeleton_tools import Skeleton, VP_type, SkeletonContainer
import networkx as nx
import math
from scipy.spatial import distance


class TestSkeletonTools(unittest.TestCase):
    def test_basicinitialization(self):
        my_identifier = 3
        my_voxel_size = np.array([1., 1., 2.])
        my_seg_id = 2
        my_nx_graph = nx.Graph()

        my_skeleton = Skeleton(identifier=my_identifier, voxel_size=my_voxel_size, seg_id=my_seg_id)
        my_skeleton_with_graph = Skeleton(identifier=my_identifier, voxel_size=my_voxel_size, seg_id=my_seg_id,
                                          nx_graph=my_nx_graph)

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
        nodes_pos_voxel = np.floor(nodes_pos_phys * scaling)
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
        self.assertEqual(my_skeleton.nx_graph.number_of_nodes(), len(edges) + 1)
        self.assertEqual(my_skeleton.nx_graph.number_of_edges(), len(edges))

    def test_addedgefeatures(self):
        my_skeleton = Skeleton()
        scaling = np.array([1., 1., 0.5])
        nodes_pos_phys = np.array([[0, 0, 0], [50, 50, 50], [100, 100, 100], [150, 150, 150]])
        nodes_pos_voxel = np.floor(nodes_pos_phys * scaling)
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

        length_voxel = np.sqrt(sum((nodes_pos_voxel[edge_u] - nodes_pos_voxel[edge_v]) ** 2))
        length_phys = np.sqrt(sum((nodes_pos_phys[edge_u] - nodes_pos_phys[edge_v]) ** 2))
        self.assertEqual(my_skeleton.nx_graph.edge[edge_u][edge_v]['length'].voxel, length_voxel)
        self.assertEqual(my_skeleton.nx_graph.edge[edge_u][edge_v]['length'].phys, length_phys)

    def test_totalPathLength(self):
        test_skeleton = Skeleton()
        nodes_pos_phys = np.array([[0, 0, 0], [5., 5., 5.], [10., 10., 10.], [15., 15., 15.]])
        edges = [(0, 1), (1, 2), (2, 3)]
        test_skeleton.initialize_from_datapoints(nodes_pos_phys, vp_type_voxel=False, edgelist=edges)

        exp_total_length = np.sqrt((15 ** 2) * 3)
        total_length = test_skeleton.calculate_total_phys_length()
        self.assertEqual(exp_total_length, total_length)

    def test_writeToKnossos(self):
        # Represents a branching skeleton. Not a real testfunction.
        # TODO add actual test function, when also reading knossos files is possible.
        test_skeleton = Skeleton(voxel_size=[10., 10., 20.])
        nodes_pos_phys = np.array([[0, 0, 0], [50., 50., 100.], [100., 100., 200.], [100., 100., 300.]])
        edges = [(0, 1), (1, 2), (1, 3)]
        test_skeleton.initialize_from_datapoints(nodes_pos_phys, vp_type_voxel=False, edgelist=edges)

        test_skeleton2 = Skeleton(voxel_size=[10., 10., 20.])
        nodes_pos_phys = np.array([[50., 50., 100.], [50., 50., 200.], [50., 50., 300.]])
        edges = [(0, 1), (1, 2)]
        test_skeleton2.initialize_from_datapoints(nodes_pos_phys, False, edgelist=edges)

        skeleton_container = SkeletonContainer([test_skeleton, test_skeleton2])
        skeleton_container.write_to_knossos_nml('testdata/knossostestfile.nml')

    def test_interpolateSkeleton(self):
        test_skeleton = Skeleton(voxel_size=[10., 10., 20.])
        nodes_pos_phys = np.array([[0, 0, 0], [50., 50., 100.], [100., 100., 200.], [100., 100., 300.]])
        edges = [(0, 1), (1, 2), (1, 3)]
        test_skeleton.initialize_from_datapoints(nodes_pos_phys, vp_type_voxel=False, edgelist=edges)

        # Get statistics about the graph and check whether they remain the same after interpolating the edges.
        # voxel
        deg = test_skeleton.nx_graph.degree()
        num_of_branch_nodes = len([n for n in deg if deg[n] > 1])
        num_of_endnodes = len([n for n in deg if deg[n] == 1])
        num_of_nodes_with_no_edge = len([n for n in deg if deg[n] == 0])

        test_skeleton.interpolate_edges(step_size=1, VP_type='voxel')
        self.assertEqual(num_of_branch_nodes, len([n for n in deg if deg[n] > 1]))
        self.assertEqual(num_of_endnodes, len([n for n in deg if deg[n] == 1]))
        self.assertEqual(num_of_nodes_with_no_edge, len([n for n in deg if deg[n] == 0]))

        # phys
        deg = test_skeleton.nx_graph.degree()
        num_of_branch_nodes = len([n for n in deg if deg[n] > 1])
        num_of_endnodes = len([n for n in deg if deg[n] == 1])
        num_of_nodes_with_no_edge = len([n for n in deg if deg[n] == 0])

        test_skeleton.interpolate_edges(step_size=1, VP_type='phys')
        self.assertEqual(num_of_branch_nodes, len([n for n in deg if deg[n] > 1]))
        self.assertEqual(num_of_endnodes, len([n for n in deg if deg[n] == 1]))
        self.assertEqual(num_of_nodes_with_no_edge, len([n for n in deg if deg[n] == 0]))

        # Test interpolation with voxel step size = 2.
        # voxel
        test_skeleton = Skeleton(voxel_size=[10., 10., 20.])
        test_skeleton.initialize_from_datapoints(nodes_pos_phys, vp_type_voxel=False, edgelist=edges)
        deg = test_skeleton.nx_graph.degree()
        num_of_branch_nodes = len([n for n in deg if deg[n] > 1])
        num_of_endnodes = len([n for n in deg if deg[n] == 1])
        num_of_nodes_with_no_edge = len([n for n in deg if deg[n] == 0])
        exp_nodes = test_skeleton.nx_graph.nodes()
        exp_edges = test_skeleton.nx_graph.edges()

        test_skeleton.interpolate_edges(step_size=2, VP_type='voxel')
        self.assertEqual(num_of_branch_nodes, len([n for n in deg if deg[n] > 1]))
        self.assertEqual(num_of_endnodes, len([n for n in deg if deg[n] == 1]))
        self.assertEqual(num_of_nodes_with_no_edge, len([n for n in deg if deg[n] == 0]))
        self.assertNotEqual(exp_nodes, test_skeleton.nx_graph.nodes())
        self.assertNotEqual(exp_edges, test_skeleton.nx_graph.edges())

        # phys
        test_skeleton = Skeleton(voxel_size=[10., 10., 20.])
        test_skeleton.initialize_from_datapoints(nodes_pos_phys, vp_type_voxel=False, edgelist=edges)
        deg = test_skeleton.nx_graph.degree()
        num_of_branch_nodes = len([n for n in deg if deg[n] > 1])
        num_of_endnodes = len([n for n in deg if deg[n] == 1])
        num_of_nodes_with_no_edge = len([n for n in deg if deg[n] == 0])
        exp_nodes = test_skeleton.nx_graph.nodes()
        exp_edges = test_skeleton.nx_graph.edges()

        test_skeleton.interpolate_edges(step_size=2, VP_type='phys')
        self.assertEqual(num_of_branch_nodes, len([n for n in deg if deg[n] > 1]))
        self.assertEqual(num_of_endnodes, len([n for n in deg if deg[n] == 1]))
        self.assertEqual(num_of_nodes_with_no_edge, len([n for n in deg if deg[n] == 0]))
        self.assertNotEqual(exp_nodes, test_skeleton.nx_graph.nodes())
        self.assertNotEqual(exp_edges, test_skeleton.nx_graph.edges())

        # Test interpolation with voxel step size too big and check that the graph does not change at all.
        # voxel
        test_skeleton = Skeleton(voxel_size=[10., 10., 20.])
        test_skeleton.initialize_from_datapoints(nodes_pos_phys, vp_type_voxel=False, edgelist=edges)
        exp_nodes = test_skeleton.nx_graph.nodes()
        exp_edges = test_skeleton.nx_graph.edges()

        test_skeleton.interpolate_edges(step_size=50, VP_type='voxel')
        self.assertEqual(exp_nodes, test_skeleton.nx_graph.nodes())
        self.assertEqual(exp_edges, test_skeleton.nx_graph.edges())

        # phys
        test_skeleton = Skeleton(voxel_size=[10., 10., 20.])
        test_skeleton.initialize_from_datapoints(nodes_pos_phys, vp_type_voxel=False, edgelist=edges)
        exp_nodes = test_skeleton.nx_graph.nodes()
        exp_edges = test_skeleton.nx_graph.edges()

        test_skeleton.interpolate_edges(step_size=200, VP_type='phys')
        self.assertEqual(exp_nodes, test_skeleton.nx_graph.nodes())
        self.assertEqual(exp_edges, test_skeleton.nx_graph.edges())

    def test_getNodeIdsOfEndpoints(self):
        # test correct number and node id of edges for branched graph
        test_skeleton_end = Skeleton(voxel_size=np.array([1.,1.,2.]))
        nodes_pos_phys_end = np.array([[0, 0, 0], [10., 5., 5.], [20., 10., 10.], [30., 15., 15.],[20.,20.,20.],[25., 25., 25.],[30., 30., 30.]])
        edges_end = [(0, 1), (1, 2), (2, 3), (1,4), (4, 5), (1, 6)]
        test_skeleton_end.initialize_from_datapoints(nodes_pos_phys_end, vp_type_voxel=False, edgelist=edges_end)

        all_endpoints = test_skeleton_end.get_node_ids_of_endpoints()
        self.assertEqual(len(all_endpoints), 4)
        self.assertEqual(set(all_endpoints), set([0,3,5,6]))

        # test correct number and node id of edges for cyclic graph
        test_skeleton_end = Skeleton(voxel_size=np.array([1.,1.,2.]))
        nodes_pos_phys_end = np.array([[0, 0, 0], [5., 5., 5.], [10., 10., 10.], [15., 15., 15.],[20.,20.,20.],[25., 25., 25.],[30., 30., 30.]])
        edges_end = [(0, 1), (1, 2), (2, 3), (3,4), (4, 5), (5, 6), (6, 0)]
        test_skeleton_end.initialize_from_datapoints(nodes_pos_phys_end, vp_type_voxel=False, edgelist=edges_end)

        all_endpoints = test_skeleton_end.get_node_ids_of_endpoints()
        self.assertEqual(len(all_endpoints), 0)
        self.assertEqual(set(all_endpoints), set([]))

        # test that does return an empty list, if only one single node is given in the connected graph
        test_skeleton_end = Skeleton(voxel_size=np.array([1.,1.,2.]))
        nodes_pos_phys_end = np.array([[5., 5., 5.]])
        test_skeleton_end.initialize_from_datapoints(nodes_pos_phys_end, vp_type_voxel=False, edgelist=None)

        all_endpoints = test_skeleton_end.get_node_ids_of_endpoints()
        self.assertEqual(len(all_endpoints), 0)

    def test_getPrecision(self):
        # test normal case where one node more predicted than in target skeleton
        Sk_target = Skeleton()
        nodes_target = np.array([[0, 0, 0], [5, 5, 10], [5, 5, 15], [20, 20, 20]])
        edges_target = ((0, 1), (1, 2), (2, 3))
        Sk_target.initialize_from_datapoints(datapoints=nodes_target, vp_type_voxel=True, edgelist=edges_target,
                                             datapoints_type='nparray')
        Sk_pred = Skeleton()
        nodes_pred = np.array([[0, 0, 5], [5, 5, 15], [10, 15, 25], [20, 25, 40]])
        edges_pred = ((0, 1), (1, 2), (2, 3))
        Sk_pred.initialize_from_datapoints(datapoints=nodes_pred, vp_type_voxel=True, edgelist=edges_pred,
                                           datapoints_type='nparray')
        tolerance_distance = 5
        human_precision = (1.+1.+0.+0.) / len(nodes_pred)
        fct_precision   = Sk_target.get_precision(other=Sk_pred, tolerance_distance=tolerance_distance)
        self.assertEqual(human_precision, fct_precision)

        # test if tolerance distance is 0, only exactly the same node position are correct
        Sk_target = Skeleton()
        nodes_target = np.array([[0, 0, 0], [5, 5, 10], [5, 5, 15], [20, 20, 20]])
        edges_target = ((0, 1), (1, 2), (2, 3))
        Sk_target.initialize_from_datapoints(datapoints=nodes_target, vp_type_voxel=True, edgelist=edges_target,
                                             datapoints_type='nparray')
        Sk_pred = Skeleton()
        nodes_pred = np.array([[0, 0, 20], [5, 5, 20], [5, 5, 15], [10, 15, 20], [20, 25, 40]])
        edges_pred = ((0, 1), (1, 2), (2, 3), (3, 4))
        Sk_pred.initialize_from_datapoints(datapoints=nodes_pred, vp_type_voxel=True, edgelist=edges_pred,
                                           datapoints_type='nparray')
        tolerance_distance = 0
        human_precision = (1.+0.+0.+0.+0.) / 5.
        fct_precision   = Sk_target.get_precision(other=Sk_pred, tolerance_distance=tolerance_distance)
        self.assertEqual(human_precision, fct_precision)

        # check if interpolated lines are taken into account
        Sk_target = Skeleton()
        nodes_target = np.array([[0, 0, 0], [20, 20, 20]])
        edges_target = [(0,1)]
        Sk_target.initialize_from_datapoints(datapoints=nodes_target, vp_type_voxel=True, edgelist=edges_target,
                                             datapoints_type='nparray')
        Sk_pred = Skeleton()
        nodes_pred = np.array([[5, 5, 5], [10, 10, 10], [30, 30, 30]])
        edges_pred = ((0, 1), (1, 2))
        Sk_pred.initialize_from_datapoints(datapoints=nodes_pred, vp_type_voxel=True, edgelist=edges_pred,
                                           datapoints_type='nparray')
        tolerance_distance = 0.
        human_precision = (1.+1.+0.) / len(nodes_pred)
        fct_precision   = Sk_target.get_precision(other=Sk_pred, tolerance_distance=tolerance_distance)
        self.assertEqual(human_precision, fct_precision)

    def test_getRecall(self):
        # normal case
        Sk_target = Skeleton()
        nodes_target = np.array([[0, 0, 0], [5, 5, 10], [5, 5, 15], [20, 20, 20]])
        edges_target = ((0, 1), (1, 2), (2, 3))
        Sk_target.initialize_from_datapoints(datapoints=nodes_target, vp_type_voxel=True, edgelist=edges_target,
                                             datapoints_type='nparray')
        Sk_pred = Skeleton()
        nodes_pred = np.array([[0, 0, 5], [5, 5, 15], [10, 15, 25], [20, 25, 40]])
        edges_pred = ((0, 1), (1, 2), (2, 3))
        Sk_pred.initialize_from_datapoints(datapoints=nodes_pred, vp_type_voxel=True, edgelist=edges_pred,
                                           datapoints_type='nparray')
        tolerance_distance = 5
        human_recall = (1.+1.+1.+0.) / len(nodes_pred)
        fct_recall   = Sk_target.get_recall(other=Sk_pred, tolerance_distance=tolerance_distance)
        self.assertEqual(human_recall, fct_recall)

        # test if tolerance distance is 0, only exactly the same node position are correct
        Sk_target = Skeleton()
        nodes_target = np.array([[0, 0, 0], [5, 5, 5], [10, 10, 10], [20, 20, 20]])
        edges_target = ((0, 1), (1, 2), (2, 3))
        Sk_target.initialize_from_datapoints(datapoints=nodes_target, vp_type_voxel=True, edgelist=edges_target,
                                             datapoints_type='nparray')
        Sk_pred = Skeleton()
        nodes_pred = np.array([[0, 0, 0], [5, 5, 6], [5, 5, 7], [10, 10, 10], [20, 20, 18]])
        edges_pred = ((0, 1), (1, 2), (2, 3), (3, 4))
        Sk_pred.initialize_from_datapoints(datapoints=nodes_pred, vp_type_voxel=True, edgelist=edges_pred,
                                           datapoints_type='nparray')
        tolerance_distance = 0
        human_precision = (1.+0.+0.+1.+0.) / 5.
        fct_precision   = Sk_target.get_precision(other=Sk_pred, tolerance_distance=tolerance_distance)
        self.assertEqual(human_precision, fct_precision)

        # check if interpolated lines are taken into account
        Sk_target = Skeleton()
        nodes_target = np.array([[0, 0, 0], [20, 20, 20]])
        edges_target = [(0, 1)]
        Sk_target.initialize_from_datapoints(datapoints=nodes_target, vp_type_voxel=True, edgelist=edges_target,
                                             datapoints_type='nparray')
        Sk_pred = Skeleton()
        nodes_pred = np.array([[0, 0, 0], [5, 5, 5], [10, 10, 10], [30, 30, 30]])
        edges_pred = ((0, 1), (1, 2), (2, 3))
        Sk_pred.initialize_from_datapoints(datapoints=nodes_pred, vp_type_voxel=True, edgelist=edges_pred,
                                           datapoints_type='nparray')
        tolerance_distance = 0.
        human_recall = (1. + 1.+ 1.) / len(nodes_pred)
        fct_recall = Sk_target.get_precision(other=Sk_pred, tolerance_distance=tolerance_distance)
        self.assertEqual(human_recall, fct_recall)

    def test_getDistanceToSkeleton(self):
        # normal case and if exactly at the same position
        Sk_target = Skeleton()
        nodes_target = np.array([[0, 0, 0], [5, 5, 10], [10, 10, 10]])
        edges_target = ((0, 1),(1, 2))
        Sk_target.initialize_from_datapoints(datapoints=nodes_target, vp_type_voxel=True, edgelist=edges_target,
                                             datapoints_type='nparray')
        nodes_pred = np.array([[0, 0, 0], [5, 5, 15]])
        true_distances = np.array([0., 5.])
        distance_to_cl = Sk_target.get_distance_to_skeleton(x=nodes_pred)
        self.assertTrue(np.array_equal(true_distances, distance_to_cl))

        # check if interpolated lines are taken into account
        Sk_target = Skeleton()
        nodes_target = np.array([[0, 0, 0], [20, 20, 20]])
        edges_target = [(0, 1)]
        Sk_target.initialize_from_datapoints(datapoints=nodes_target, vp_type_voxel=True, edgelist=edges_target,
                                             datapoints_type='nparray')
        nodes_pred = np.array([[0, 0, 0], [5, 5, 5], [10, 10, 10]])

        true_distances = np.array([0., 0., 0.])
        distance_to_cl = Sk_target.get_distance_to_skeleton(x=nodes_pred)
        self.assertTrue(np.array_equal(true_distances, distance_to_cl))

    def test_applyTransformation(self):

        try:
            import augment
        except ImportError:
            return

        # normal case and if exactly at the same position
        test_skeleton = Skeleton()
        nodes_target = np.array([[0, 0, 0], [1, 0, 1], [2, 1, 2]])
        edges_target = ((0, 1),(1, 2))
        test_skeleton.initialize_from_datapoints(datapoints=nodes_target, vp_type_voxel=True, edgelist=edges_target,
                                             datapoints_type='nparray')

        nodes_bb = (3, 3, 3)
        # Create transformation matrix.
        transformation = augment.create_identity_transformation(nodes_bb)

        # Rotate around z axis 90 degree anti-clockwise.
        transformation += augment.create_rotation_transformation(
            nodes_bb,
            math.pi/2)

        test_skeleton.apply_transformation(transformation)
        pred_pos0 = test_skeleton.nx_graph.node[0]['position'].voxel
        pred_pos1 = test_skeleton.nx_graph.node[1]['position'].voxel
        pred_pos2 = test_skeleton.nx_graph.node[2]['position'].voxel

        exp_pos0 = np.array([2, 0, 0])
        exp_pos1 = np.array([2, 1, 1])
        exp_pos2 = np.array([1, 2, 2])

        self.assertTrue((pred_pos0 == exp_pos0).all())
        self.assertTrue((pred_pos1 == exp_pos1).all())
        self.assertTrue((pred_pos2 == exp_pos2).all())


    def test_ShiftSkeleton(self):
        Sk = Skeleton()
        nodes = np.array([[0, 0, 0], [5, 5, 10], [10, 10, 10]])
        edges = ((0, 1),(1, 2))
        Sk.initialize_from_datapoints(datapoints=nodes, vp_type_voxel=True, edgelist=edges,
                                             datapoints_type='nparray')

        offset_1 = np.asarray([1,1,1])
        new_pos0 = nodes[0] + offset_1
        Sk.shift_skeleton(offset=offset_1, VP_type='voxel')
        pos0 = Sk.nx_graph.node[0]['position'].voxel
        self.assertTrue(np.array_equal(new_pos0, pos0))
        self.assertEqual(None, Sk.nx_graph.node[0]['position'].phys)


        Sk = Skeleton()
        nodes = np.array([[0, 0, 0], [5, 5, 10], [10, 10, 10]])
        edges = ((0, 1),(1, 2))
        Sk.initialize_from_datapoints(datapoints=nodes, vp_type_voxel=False, edgelist=edges,
                                             datapoints_type='nparray')
        offset_2 = np.asarray([1,2,3])
        new_pos0 = nodes[0] + offset_2
        Sk.shift_skeleton(offset=offset_2, VP_type='phys')
        pos0 = Sk.nx_graph.node[0]['position'].phys
        self.assertTrue(np.array_equal(new_pos0, pos0))
        self.assertEqual(None, Sk.nx_graph.node[0]['position'].voxel)

    def test_CheckPointInsideBb(self):
        Sk = Skeleton()
        nodes = np.array([[0, 0, 0], [5, 5, 5], [6,6,6], [8,8,8]])
        edges = ((0, 1),(1, 2),(2,3))
        Sk.initialize_from_datapoints(datapoints=nodes, vp_type_voxel=True, edgelist=edges,
                                             datapoints_type='nparray')

        bb_min = [1,1,1]
        bb_max = [6,6,6]
        # check points between min und max and exactly min and max are still counted as inside
        self.assertTrue(Sk.check_point_inside_bb([1,1,1], bb_min=bb_min, bb_max=bb_max))
        self.assertTrue(Sk.check_point_inside_bb([4,4,4], bb_min=bb_min, bb_max=bb_max))
        self.assertTrue(Sk.check_point_inside_bb([6,6,6], bb_min=bb_min, bb_max=bb_max))

        # check points smaller than min and larger than max are counted as outside
        self.assertFalse(Sk.check_point_inside_bb([0,0,0], bb_min=bb_min, bb_max=bb_max))
        self.assertFalse(Sk.check_point_inside_bb([7,7,7], bb_min=bb_min, bb_max=bb_max))

    def test_CropGraphToBb(self):
        bb_min = [1,1,1]
        bb_max = [6,6,6]

        Sk = Skeleton()
        nodes = np.array([[0, 0, 0], [5, 5, 5], [6,6,6], [8,8,8]])
        num_nodes_outside = 2
        edges = ((0, 1),(1, 2),(2,3))
        Sk.initialize_from_datapoints(datapoints=nodes, vp_type_voxel=True, edgelist=edges,
                                             datapoints_type='nparray')

        Sk2 = Skeleton()
        nodes2 = np.array([[1, 1, 1], [5, 5, 5], [6,6,6]])
        edges2 = ((0, 1),(1, 2))
        Sk2.initialize_from_datapoints(datapoints=nodes2, vp_type_voxel=True, edgelist=edges2,
                                             datapoints_type='nparray')

        # example where 2 nodes outside
        Sk.crop_graph_to_bb(bb_min, bb_max)
        for node_id in Sk.nx_graph.nodes_iter():
            self.assertTrue(Sk.check_point_inside_bb(Sk.nx_graph.node[node_id]['position'].voxel, bb_min, bb_max))
        self.assertEqual(Sk.nx_graph.number_of_nodes(), len(nodes)-num_nodes_outside)

        # example with no nodes outside
        Sk2.crop_graph_to_bb(bb_min, bb_max)
        for node_id in Sk2.nx_graph.nodes_iter():
            self.assertTrue(Sk2.check_point_inside_bb(Sk2.nx_graph.node[node_id]['position'].voxel, bb_min, bb_max))

        self.assertEqual(Sk2.nx_graph.number_of_nodes(), len(nodes2))

    def test_GetSegIDs(self):
        segmentation = np.zeros((20, 20, 20))
        segmentation[:, :, 0:7] = 1
        segmentation[:, :, 7:10] = 2
        segmentation[:, :, 10:16] = 3
        segmentation[:, :, 16:20] = 4

        Sk = Skeleton()
        nodes = np.array([[10, 10, 0], [10, 10, 15]])
        edges = [[0, 1]]
        Sk.initialize_from_datapoints(datapoints=nodes, vp_type_voxel=True, edgelist=edges,
                                      datapoints_type='nparray')

        Sk.interpolate_edges()
        # Test basics, collect all seg_ids.
        seg_ids, object_dict = Sk.get_seg_ids(segmentation, size_thres=0, return_objectdict=True)
        expected_segids = np.array([1, 2, 3])
        self.assertTrue(set(expected_segids) == set(seg_ids))
        self.assertEqual(7, object_dict[1]['size'])

        # Test size_thres.
        seg_ids, object_dict = Sk.get_seg_ids(segmentation, size_thres=4, return_objectdict=True)
        expected_segids = np.array([1, 3])
        self.assertTrue(set(expected_segids) == set(seg_ids))
        self.assertEqual(3, object_dict[2]['size'])

        # Test returning empty array if skeleton coordinates are outside the cube.
        Sk.shift_skeleton([20, 20, 20], 'voxel')
        seg_ids, object_dict = Sk.get_seg_ids(segmentation, size_thres=0, return_objectdict=True)
        self.assertEqual(0, len(list(seg_ids)))
        self.assertEqual(0, len(object_dict))

    def test_SkContainerSplitIntoCCs(self):
        # Represents a branching skeleton.
        test_skeleton = Skeleton()
        nodes_pos = np.array([[5, 0, 5], [5, 5, 5], [10, 10, 10], [15, 10, 8]])
        edges = [(0, 1), (1, 2), (1, 3)]
        test_skeleton.initialize_from_datapoints(nodes_pos, vp_type_voxel=True, edgelist=edges)
        test_skeleton.interpolate_edges()
        skeleton_container = SkeletonContainer([test_skeleton])

        for skeleton in skeleton_container.skeleton_list:
            for node_id, node_dic in skeleton.nx_graph.nodes_iter(data=True):
                print node_id, node_dic['position'].voxel
            print skeleton.nx_graph.edges()


        for skeleton in skeleton_container.skeleton_list:
            skeleton.crop_graph_to_bb([6, 6, 6], [20, 20, 20])

        self.assertEqual(1, len(skeleton_container.skeleton_list))
        skeleton_container.split_into_cc()

        self.assertEqual(2, len(skeleton_container.skeleton_list))










    def test_CheckPointInsideBb(self):
        Sk = Skeleton()
        nodes = np.array([[0, 0, 0], [5, 5, 5], [6,6,6], [8,8,8]])
        edges = ((0, 1),(1, 2),(2,3))
        Sk.initialize_from_datapoints(datapoints=nodes, vp_type_voxel=True, edgelist=edges,
                                             datapoints_type='nparray')

        bb_min = [1,1,1]
        bb_max = [6,6,6]
        # check points between min und max and exactly min and max are still counted as inside
        self.assertTrue(Sk.check_point_inside_bb([1,1,1], bb_min=bb_min, bb_max=bb_max))
        self.assertTrue(Sk.check_point_inside_bb([4,4,4], bb_min=bb_min, bb_max=bb_max))
        self.assertTrue(Sk.check_point_inside_bb([5,5,5], bb_min=bb_min, bb_max=bb_max))

        # check points smaller than min and larger than max are counted as outside
        self.assertFalse(Sk.check_point_inside_bb([0,0,0], bb_min=bb_min, bb_max=bb_max))
        self.assertFalse(Sk.check_point_inside_bb([7,7,7], bb_min=bb_min, bb_max=bb_max))

    def test_CropGraphToBb(self):
        bb_min = [1,1,1]
        bb_max = [7,7,7]

        Sk = Skeleton()
        nodes = np.array([[0, 0, 0], [5, 5, 5], [6,6,6], [8,8,8]])
        num_nodes_outside = 2
        edges = ((0, 1),(1, 2),(2,3))
        Sk.initialize_from_datapoints(datapoints=nodes, vp_type_voxel=True, edgelist=edges,
                                             datapoints_type='nparray')

        Sk2 = Skeleton()
        nodes2 = np.array([[1, 1, 1], [5, 5, 5], [6,6,6]])
        edges2 = ((0, 1),(1, 2))
        Sk2.initialize_from_datapoints(datapoints=nodes2, vp_type_voxel=True, edgelist=edges2,
                                             datapoints_type='nparray')

        # example where 2 nodes outside
        Sk.crop_graph_to_bb(bb_min, bb_max)
        for node_id in Sk.nx_graph.nodes_iter():
            self.assertTrue(Sk.check_point_inside_bb(Sk.nx_graph.node[node_id]['position'].voxel, bb_min, bb_max))
        self.assertEqual(Sk.nx_graph.number_of_nodes(), len(nodes)-num_nodes_outside)

        # example with no nodes outside
        Sk2.crop_graph_to_bb(bb_min, bb_max)
        for node_id in Sk2.nx_graph.nodes_iter():
            self.assertTrue(Sk2.check_point_inside_bb(Sk2.nx_graph.node[node_id]['position'].voxel, bb_min, bb_max))

        self.assertEqual(Sk2.nx_graph.number_of_nodes(), len(nodes2))