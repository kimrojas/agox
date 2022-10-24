import numbers
import os
import joblib
import ray

ray_kwarg_keys = ('tmp_dir', 'memory', 'cpu_count')

class RayBaseClass:

    def __init__(self, tmp_dir=None, memory=None, cpu_count=None):
        self.tmp_dir = tmp_dir
        self.memory = memory
        self.cpu_count = cpu_count

    def ray_startup(self):
        """
        Start a Ray instance on Slurm.

        Parameters
        ----------
        cpu_count : int, optional
            Number of CPU cores to use, by default None in which case defaults are 
            read from SLURM.
        memory : int, optional
            Amount of memeory to use in bits, by default None in which case environment
            variables are used to guess a suitable amount. 
        """
        cpu_count = self.cpu_count
        memory = self.memory 
        tmp_dir = self.tmp_dir

        # CPU Count:
        if cpu_count is None:
            try:
                cpu_count = int(os.environ['SLURM_NTASKS'])
            except:
                cpu_count = joblib.cpu_count()
        # Memory:
        if memory is None:
            try:
                memory=int(os.environ['SLURM_MEM_PER_NODE']*1e6)
            except:
                memory=cpu_count*int(2*1e9)

        if tmp_dir is None:
            path = os.getcwd()
            tmp_dir = os.path.join(path, 'ray')

            number_of_bytes = len(tmp_dir.encode('utf-8')) + 61

            if number_of_bytes > 107:
                tmp_dir = os.path.expanduser('~') + '/tmp/ray'
                print('USING USER ROOT FOLDER FOR RAY TEMP BECAUSE GIVEN OR DEFAULT GENERATED PATH IS TOO MANY BYTES.')
                print(f'Path: {tmp_dir}')
        
        if not ray.is_initialized():
            ray.init(_memory=memory, object_store_memory=int(memory/4),
                     num_cpus=cpu_count, ignore_reinit_error=True, _temp_dir=tmp_dir)
            print(ray.cluster_resources())
