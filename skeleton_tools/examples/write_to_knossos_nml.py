
from skeleton_tools.skeleton_tools import SkeletonContainer

inputfilename = '/raid/julia/data/segem/SegEM_challenge/skeletonData/cortex_training.nml'
outputfilename = '/raid/julia/projects/LSTM/reni/segem/test.nml'

sk_con = SkeletonContainer()
sk_con.read_from_knossos_nml(inputfilename)
sk_con.write_to_knossos_nml(outputfilename)



