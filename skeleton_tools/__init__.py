try:
    from skeleton_tools import Skeleton
    from skeleton_tools import SkeletonContainer
    from skeleton_utils import evaluate_segmentation_with_gt_skeletons, replace
    from skeleton_utils import read_from_pickle, write_to_pickle
except:
    ImportError

try:
    from .skeleton_tools import Skeleton
    from .skeleton_tools import SkeletonContainer
    from .skeleton_utils import evaluate_segmentation_with_gt_skeletons, replace
    # from .skeleton_utils import read_from_pickle, write_to_pickle
except:
    ImportError


