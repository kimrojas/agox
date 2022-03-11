import numpy as np
from agox.modules.helpers.grid_imitator import GridImitator
from agox.modules.postprocessors.postprocess_ABC import PostprocessBaseClass

class SnapperPostProcess(PostprocessBaseClass):

    name = 'Snappy'

    def __init__(self, grid_imitator, **kwargs):
        super().__init__(**kwargs)
        self.grid_imitator = grid_imitator        
        assert self.grid_imitator.__class__ == GridImitator # Has to be this imitator, cannot be a real grid.

    @PostprocessBaseClass.immunity_decorator
    def postprocess(self, candidate):

        actions = self.grid_imitator.xyz_to_ijk(candidate.get_positions(), snap=True)
        atom_types = candidate.get_atomic_numbers()

        # Check that no two actions are the same. If they are, move one of them.
        # FIND A BETTER SOLUTION
        while not len(actions)==len(np.unique(actions,axis=0)):
            # Locate of the action that are causing problems
            unique_actions,count = np.unique(actions,axis=0,return_counts=True)
            offending_actions = unique_actions[count>1][0]
            index_of_offending_action = np.where((actions==offending_actions).all(1))[0][-1]
            # Adjust this action
            actions[index_of_offending_action]+=np.array([1,0,0])

        #new_grid = self.grid.copy()

        # Check that all atoms still on grid
        for action in actions[len(candidate.template):]:
            if np.any(action < 0) or np.any(action >= self.grid_imitator.grid_shape):
                print('violation',action,self.grid_imitator.grid_shape)
                return None
                
        #for action,a in zip(actions[len(self.grid):],atom_types[len(self.grid):]):
        #    new_grid.add_atom([action[0],action[1],action[2]],atom_type = a)

        c = len(candidate.get_template())
        for action in actions[len(candidate.template):]:
            xyz = self.grid_imitator.ijk_to_xyz(action)
            candidate.positions[c] = xyz
            c += 1
    
        return candidate
