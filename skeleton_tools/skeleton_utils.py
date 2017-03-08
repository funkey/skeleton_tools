import numpy as np
import networkx as nx
from networkx.algorithms import bipartite



def replace(arr, rep_dict):
    """
    Creates a new segmentation based on a replacement dictionary.
    Parameters
    ----------
        arr:  ndarray,
            an array with a single ID per supervoxel / connected component.
        rep_dict: dictionary
            Keys correspond to supervoxel ID, value to the new ID. (Assumes all elements of "arr" are keys of rep_dict)
     Returns
     ---------
        ndarray
            New array with the same shape as input array and with the values of the replace_dict as new IDs.
            """

    rep_keys, rep_vals = np.array(list(zip(*sorted(rep_dict.items()))))
    idces = np.digitize(arr, rep_keys, right=True)
    return rep_vals[idces]


def evaluate_segmentation_with_gt_skeletons(segmentation, sk_container,
                                            return_new_seg=False, size_thres=0):
    """
    Calculates merges and splits for a given segmentation based on provided skeletons.

    Parameters
    ----------
        segmentation:  ndarray,
            3D volume with a single ID per supervoxel / connected component.
        sk_container: skeleton_tools.SkeletonContainer
            The ground truth skeletons.
        return_new_seg: bool
            If set to true, a new array is created in which the supervoxels are merged based on the
            ground truth skeletons. Background is 0.
        size_thres: int
            Only consider segments in which the skeleton has more than size_thres number of nodes.
     Returns
     ---------
        num_of_merges: int
            Note that if a supervoxel contains 3 skeleton object, this counts as 2 merges.
        num_of_splits: int
        SVids_merger:
        bipart_graph: the graph that was created for evaluation. Node set 1: skeleton objects, node set 2:
        supervoxel IDs.

     Notes
     --------
        Volume needs to be denseley skeletonized (otherwise merge and splits are misleading).
    """
    all_ids_covered = []
    replace_dict = {}  # to obtain a new segmentation based on the skeletons.# Key: original SV ID, value: new seg_id.
    skel_dic = {}
    for ii, skeleton in enumerate(sk_container.skeleton_list):
        skeleton_id = ii + 1
        id_list, object_dict = skeleton.get_seg_ids(segmentation, size_thres=size_thres, return_objectdict=True)
        all_ids_covered.extend(id_list)
        skel_dic[skeleton_id] = id_list
        for id in id_list:
            replace_dict.update({id: skeleton_id})

    # Build a bipartite graph. One set of nodes correspond to skeletons, the other set to Supervoxels.
    # Number of Mergers --> Number of skeleton_ids that are associated with the same supervoxel.
    # Number of Splits --> Number of Supervoxels that are associated with the same skeleton.
    bipart_graph = nx.Graph()
    # Add skeleton ids (convert to hex to make them different type versus supervoxel Ids).
    bipart_graph.add_nodes_from([hex(skeleton_id) for skeleton_id in skel_dic.keys()], bipartite=0)
    sv_nodes = np.unique(segmentation)
    bipart_graph.add_nodes_from(sv_nodes, bipartite=1)
    edges = []
    for skeleton_id, SVids in skel_dic.iteritems():
        # Remove background ID:
        SVids = list(SVids)
        if 0 in SVids:
            SVids.remove(0)
        for SVid in SVids:
            edges.append([hex(skeleton_id), SVid])

    bipart_graph.add_edges_from(edges)

    num_of_merges, num_of_splits, skeleton_groups, SVids_merger = from_bipartitegraph_to_mergesplitmetric(bipart_graph,
                                                                                                          sv_nodes)
    if return_new_seg:
        # Create new segmentation.
        # replace_dict so far maps SVids to skeleton_ids. Find all those skeletons, that are merged, and replace merged
        # skeleton_ids with new ones.
        # --> Find connected components in the skeleton_groups (use union find data structure for that).
        union_datast = nx.utils.union_find.UnionFind()
        for skeleton_group in skeleton_groups:
            union_datast.union(*skeleton_group)

        skeleton_to_parent = union_datast.parents
        for SV_id, skeleton_id in replace_dict.iteritems():
            if skeleton_id in skeleton_to_parent:
                replace_dict.update({SV_id: skeleton_to_parent[skeleton_id]})
        all_ids = set(np.unique(segmentation))
        all_ids_not_covered = all_ids - set(all_ids_covered)
        # All SVS that did not contain any skeletons should get the background color: 0.
        for id in all_ids_not_covered:
            replace_dict.update({id: 0})

        replace_dict[0] = 0
        new_segmentation = replace(segmentation, replace_dict).astype(np.uint64)
        return num_of_merges, num_of_splits, SVids_merger, bipart_graph, new_segmentation
    else:
        return num_of_merges, num_of_splits, SVids_merger, bipart_graph


def from_bipartitegraph_to_mergesplitmetric(bipart_graph, sv_nodes):
    degX, degY = bipartite.degrees(bipart_graph, sv_nodes)
    num_of_merges = 0
    SVids_merger = []
    skeleton_groups = []
    for node_id, node_degree in degY.iteritems():
        if node_degree > 1:
            num_of_merges += node_degree - 1
            SVids_merger.append(node_id)
            skeletons = bipart_graph[node_id].keys()  # This gets all the nodes the node is connected to.
            skeletons = [int(id, 16) for id in skeletons]
            skeleton_groups.append(skeletons)

    num_of_splits = 0
    for node_id, node_degree in degX.iteritems():
        if node_degree > 1:
            num_of_splits += node_degree - 1
    return num_of_merges, num_of_splits, skeleton_groups, SVids_merger





