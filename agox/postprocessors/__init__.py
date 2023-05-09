"""
While generators generate candidates, postprocessors are used to apply common 
postprocessing steps to all generated candidates. For example, the centering postprocessor
will center all candidates in the cell. 

The most common postprocessors is the relax postprocessor, which performs a local
optimization on all candidates. Parallel implementations of this postprocessor are
also available and are recommended for searches that generated multiple candidates 
pr. iteration. 
"""
from .ABC_postprocess import PostprocessBaseClass
from .wrap import WrapperPostprocess
from .centering import CenteringPostProcess
from .mpi_relax import MPIRelaxPostprocess
from .relax import RelaxPostprocess
from agox.postprocessors.ray_relax import ParallelRelaxPostprocess as ParallelRemoteRelaxPostprocess
from agox.postprocessors.ray_pool_relax import ParallelRelaxPostprocess