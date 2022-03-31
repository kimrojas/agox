import numpy as np

class GridImitator:

    """
    This exists to give certain modules access to methods that help deal with discrete/grid coordinates - this is used 
    rather than the ASLA Grid-object to avoid extra complexity with assign_from functions
    """

    def __init__(self, grid_anchor=None, size=None, scale=None, grid=None):
        if grid is not None:
            self.grid_anchor = grid.grid_anchor
            self.size = grid.size
            self.scale = grid.scale
            self.grid_shape = grid.grid_shape
        else:
            assert (grid_anchor is not None) * (size is not None) * (scale is not None)
            self.grid_anchor = grid_anchor
            self.size = size
            self.scale = scale
            self.grid_shape = [max(int(x / self.scale + 0.01), 1) for x in self.size]

    def ijk_to_xyz(self, ijk, padding=[0, 0, 0], relative='cell'):
        """
        Takes one or more sets of grid indices and returns the
        corresponding cartesian coordinates.
        """
        # Save input shape for use at return
        input_shape = np.array(ijk).shape

        # input ijk [0,0,0] -> [[0,0,0]]
        if len(input_shape) == 1:
            ijk = [ijk]
            
        ijk = np.array(ijk, dtype = 'float64')

        # Subtract padding
        ijk -= padding

        # Apply scaling
        xyz = ijk * self.scale

        if relative == 'cell':
            # Add anchor point
            xyz += np.array(self.grid_anchor) 

        # If a single grid point is passed, return the coords as
        # [x,y,z] instead of [[x,y,z]]
        if len(input_shape) == 1:
            return xyz.flatten()
        else:
            return xyz

    def xyz_to_ijk(self, xyz, padding=[0, 0, 0], relative='cell', snap=False):
        """ 
        Takes one or more sets of cartesian coordinates and returns
        the ijk coordinates of the nearest grid point. If snap is
        True, the xyz are snapped to nearest grid point.
        """
        c = xyz.copy()
        if len(c.shape) == 1:
            c = c.reshape(-1,3)

        if relative=='cell':
            # Subtract anchorpoint
            c -= np.array(self.grid_anchor)
            
        # Apply inverse scaling
        g = c / self.scale

        # Add padding for x, y and z
        g += padding
        
        # Round to nearest grid point
        if snap:
            g = np.round(g).astype(int)

        return g
