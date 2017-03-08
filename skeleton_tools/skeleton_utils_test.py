import numpy as np
import unittest
from skeleton_tools import Skeleton, SkeletonContainer
from skeleton_utils import evaluate_segmentation_with_gt_skeletons



class TestSkeletonUtils(unittest.TestCase):
    def test_evaluate_segmentation_with_gt_skeletons(self):
        segmentation = np.zeros((20, 20, 20))
        segmentation[:, :, 0:7] = 1
        segmentation[:, :, 7:10] = 2
        segmentation[:, :, 10:16] = 3
        segmentation[:, :, 16:20] = 4
        segmentation[:, 0:5, : 16] = 5

        sk_left = Skeleton()
        nodes = np.array([[10, 3, 0], [10, 3, 20]])  # skeleton that passes y-left part of the cube (ID 5 and 4)
        edges = [[0, 1]]
        sk_left.initialize_from_datapoints(datapoints=nodes, vp_type_voxel=True, edgelist=edges,
                                           datapoints_type='nparray')
        sk_left.interpolate_edges()

        sk_right = Skeleton()
        nodes = np.array([[10, 7, 0], [10, 7, 20]])  # skeleton that passes y-right part of the cube (ID 1 2 3 4 5)
        edges = [[0, 1]]
        sk_right.initialize_from_datapoints(datapoints=nodes, vp_type_voxel=True, edgelist=edges,
                                            datapoints_type='nparray')
        sk_right.interpolate_edges()
        sk_con = SkeletonContainer(skeletons=[sk_left, sk_right])
        num_merges, num_splits, SV_merges, bigraph, new_seg = evaluate_segmentation_with_gt_skeletons(segmentation,
                                                                                     sk_con,
                                                                                     return_new_seg=True,
                                                                                     size_thres=0)

        expected_SV_merger = [5]
        self.assertEqual(1, num_merges)
        self.assertEqual(4, num_splits)
        self.assertTrue(set(expected_SV_merger), SV_merges)
        # Since the two skeletons have a common supervoxel, the resulting new segmentation should have exactly
        # one connected component.
        self.assertEqual(1, len(np.unique(new_seg)))


