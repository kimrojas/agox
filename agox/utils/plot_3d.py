import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tarfile
import shutil

from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, CheckButtons

from ase.io import read
from ase.data.colors import jmol_colors
from ase.data import covalent_radii

from argparse import ArgumentParser


class MainPlot:

    def __init__(self, atoms, Q_values, grid_anchor, grid_scale):
        '''
        Plotting module for Q-values
        '''

        self.atoms = atoms
        self._transform_atoms(grid_anchor, grid_scale)
        self.Q_values = Q_values # [Steps, X, Y, Z, species]
        self.n_steps = Q_values.shape[0]
        self._step = 0
        self._atom_types = sorted(list(set(atoms.get_atomic_numbers())))
        
        self.n_species = self.Q_values.shape[-1]

        self.fig, self.axes = plt.subplots(nrows = 1,
                                           ncols = self.n_species,
                                           figsize = (7 * self.n_species, 7),
                                           subplot_kw = {'projection':'3d'})

        plt.subplots_adjust(left = 0,
                            right = 1,
                            top = 1,
                            bottom = 0,
                            wspace = 0,
                            hspace = 0)
        
        self.subplots = []

        for i in range(self.Q_values.shape[-1]):
            subplot = SubPlot(main_plot = self,
                              fig = self.fig,
                              ax = self.axes[i],
                              atoms = atoms,
                              Q = self.Q_values[:,:,:,:,i],
                              grid_scale = grid_scale,
                              atom_type = self._atom_types[i])
            self.subplots.append(subplot)

        self._plot_step()
        self._plot_button()
        self.fig.canvas.mpl_connect('key_press_event', self.keypress_step)
        
    def _plot_step(self):
        self.step_plot = self.axes[0].annotate(f'Step:{self.step}',
                                               xy = (0.1, 0.85),
                                               xycoords = 'axes fraction')

    def _update_step(self):
        self.step_plot.set_text(f'Step:{self.step}')

    def _plot_button(self):
        ax_button = self.fig.add_axes([0.1 /self.n_species, 0.6, 0.15 / self.n_species, 0.15])
        ax_button.set_axis_off()
        self._button = CheckButtons(ax_button,
                                    labels = ['Show highest', 'Show top 30'],
                                    actives = [False, False])
        self._button.on_clicked(self._update_button)

    def _update_button(self, label):
        if label == 'Show highest':
            self._update_highest_Q()
        elif label == 'Show top 30':
            self._update_top_30()

    def _update_highest_Q(self):
        mark = self._button.get_status()[0] # First button is 'Show highest'
        Q_max_idx = np.argmax(self.Q_values[self.step])
        X,Y,Z, T = np.unravel_index(Q_max_idx, self.Q_values[0].shape)

        if mark:
            color = 'red'
        else:
            qval = self.Q_values[self.step, X, Y, Z, T]
            color = self.subplots[T]._get_colors(qval)

        try:
            voxel = self.subplots[T].voxels[(X,Y,Z)]
            voxel.set_facecolor(color) # Not ideal though. Need to replot instead.
        except KeyError:
            pass # Voxel is not visible
        self.fig.canvas.draw_idle()
        
    def _update_top_30(self):
        status = self._button.get_status()[1] # Second button is 'Top 30'
        for subplot in self.subplots:
            if status is False:
                subplot.update_Qvalues()
            else:
                subplot.update_top_30() 
        
    def _transform_atoms(self, grid_anchor, grid_scale):
        # First adjust such that grid_anchor is at [0,0,0]
        self.atoms.positions -= grid_anchor

        # Then scale positions since 1 voxel is 1 x 1 x 1
        self.atoms.positions /= grid_scale

        # Then move positions 0.5 pixel, since voxels start in [0,0,0], which means
        # first position should be at [0.5, 0.5, 0,5]
        self.atoms.positions += [0.5, 0.5, 0.5]

    def keypress_step(self, event):

        if event.key == 'right':
            self.step += 1
        if event.key == 'left':
            self.step -= 1

        if event.key == 'right' or event.key == 'left':

            # Update subplot information
            for subplot in self.subplots:
                subplot.update_Qvalues()
                subplot.update_atoms()

            # Update global information
            self._update_step()
            self._update_highest_Q()
            
    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, new):
        if new == self.Q_values.shape[0]:
            print('No Q-values for final step')
        elif new < 0:
            print('Too few atoms placed')
        else:
            self._step = new

                
class SubPlot:
    '''
    Module for plotting 3D Qvalues for 1 species
    '''
    
    def __init__(self, main_plot, fig, ax, atoms, Q, grid_scale, atom_type):
        """
        
        Parameters
        ----------
        self : 

        **kwargs : 


        Returns
        -------
        out : 

        """

        self.main_plot = main_plot
        self.fig = fig
        self.ax = ax
        self.grid_scale = grid_scale
        self.spheres = []
        self.atom_type = atom_type
        
        self.ax.set_xlim([0, Q.shape[1]])
        self.ax.set_ylim([0, Q.shape[2]])
        self.ax.set_zlim([0, Q.shape[3]])
        self._set_axes_equal()

        self.ax.set_axis_off()
        
        self.atoms = atoms
        
        self.Q = Q
        self.Q_low = 0.5
        self.Q_high = 1.5

        self.voxels = {}
        self.update_Qvalues()
        self.plot_atoms()
        self.plot_type()
        
        self.ax_slider_low, self.ax_slider_high = self._get_slider_axes()
        
        self.slider_low = Slider(self.ax_slider_low, 'Q-low', -1.5, 1.5, valinit = self.Q_low, valstep = 0.1)
        self.slider_high = Slider(self.ax_slider_high, 'Q-high', -1.5, 1.5, valinit = self.Q_high, valstep = 0.1)
        
        self.slider_low.on_changed(self.update_Qvalues_low)        
        self.slider_high.on_changed(self.update_Qvalues_high)

    def plot_type(self):
        self.ax.annotate(f'Atom-type:{self.atom_type}',
                         xy = (0.1, 0.8),
                         xycoords = 'axes fraction')        

    def _get_slider_axes(self):
        bbox1 = self.ax.get_position().bounds # x0,y0, width, height
        bbox1 = Bbox([[bbox1[0] + 0.05, bbox1[1] + 0.05],
                      [bbox1[0] + bbox1[2] - 0.1, bbox1[1] + 0.1]])
        ax1 = plt.axes(bbox1)

        bbox2 = self.ax.get_position().bounds # x0,y0, width, height
        bbox2 = Bbox([[bbox2[0] + 0.05, bbox2[1] + 0.1],
                      [bbox2[0] + bbox2[2] - 0.1, bbox2[1] + 0.15]])
        
        ax2 = plt.axes(bbox2)

        return ax1, ax2

    def _get_colors(self, Q):
        norm = Normalize(vmin = self.Q_low, vmax = self.Q_high)
        return cm.Blues(norm(Q))

    def plot_atoms(self):
        positions = self.atoms.get_positions() # [:-self.Q.shape[0] + self.main_plot.step]
        numbers = self.atoms.get_atomic_numbers() # [:-self.Q.shape[0] + self.main_plot.step]

        for i in range(positions.shape[0]):
            sphere = self.plot_sphere(positions[i], numbers[i])
            self.spheres.append(sphere)

        for sphere in self.spheres[-self.Q.shape[0] + self.main_plot.step:]:
            sphere.set_visible(False)
        
    def plot_sphere(self, pos, atom_type = 1, points = 10):
        radius = covalent_radii[atom_type] * 0.9 / self.grid_scale
        color = jmol_colors[atom_type]
        u, v = np.linspace(0, 2 * np.pi, points), np.linspace(0, np.pi * points)
        x = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]

        sphere = self.ax.plot_surface(x, y, z, color = color)

        return sphere
        
    def update_Qvalues_high(self, val):
        self.Q_high = val
        self.update_Qvalues()
        self.main_plot._update_highest_Q()

    def update_Qvalues_low(self, val):
        self.Q_low = val
        self.update_Qvalues()
        self.main_plot._update_highest_Q()

    def update_Qvalues(self):
        '''
        We redraw all the voxels, as the inner parts are not plotted.
        '''

        print('Min Q:', np.min(self.Q[self.main_plot.step]))        
        print('Max Q:', np.max(self.Q[self.main_plot.step]))

        # First we remove all the old voxels
        for voxel in self.voxels.values():
            voxel.remove()

        fill = np.ones(self.Q[self.main_plot.step].shape, dtype = np.bool)

        # Remove low and high Q-values
        fill[self.Q[self.main_plot.step] < self.Q_low] = False
        fill[self.Q[self.main_plot.step] > self.Q_high] = False
        
        colors = self._get_colors(self.Q[self.main_plot.step])

        self.voxels = self.ax.voxels(fill, facecolors = colors)

        self.fig.canvas.draw_idle()

    def update_atoms(self):
        '''
        Hide/show atoms depending on step parameter
        
        '''
        # First make them all visible
        for sphere in self.spheres:
            sphere.set_visible(True)

        # Then hide the future ones
        for sphere in self.spheres[-self.Q.shape[0] + self.main_plot.step:]:
            sphere.set_visible(False)

    def update_top_30(self):
        # First remove Q-values below Q-low
        Q_val = np.sort(self.Q[self.main_plot.step].flatten())[-30]

        if Q_val < self.Q_low:
            return
        
        out = np.vstack(np.where(self.Q[self.main_plot.step] > Q_val)).T
        out = [tuple(x) for x in out.tolist()]
        for ijk in out:
            try:
                self.voxels[ijk].set_facecolor('red')
            except KeyError:
                pass
                # Voxel is not visible
        
        # Q_max_idx = np.argmax(self.Q_values[self.step])
        # X,Y,Z, T = np.unravel_index(Q_max_idx, self.Q_values[0].shape)
        #     color = 'red'
        # else:
        #     qval = self.Q_values[self.step, X, Y, Z, T]
        #     color = self.subplots[T]._get_colors(qval)
            
        # voxel = self.subplots[T].voxels[(X,Y,Z)]
        # voxel.set_facecolor(color) # Not ideal though. Need to replot instead.
        self.fig.canvas.draw_idle()            

    def _hide_voxels(self):
        '''
        Hide voxels not in Q-value range. 
        Unfortunately, matplotlib does not plot the inner part of the voxels. 
        So does not work, untill matplotlib changes.
        '''
        out = np.vstack(np.where((self.Q[self.main_plot.step] < self.Q_low) | (self.Q[self.main_plot.step] > self.Q_high))).T
        out = [tuple(x) for x in out.tolist()]

        for ijk in out:
            self.voxels[ijk].set_visible(False)

    def _set_axes_equal(self):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        See
        https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to


        '''

        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        
if __name__ == '__main__':

    # Input arguments:
    parser = ArgumentParser()
    parser.add_argument('tarfile', type=str)

    args = parser.parse_args()
    tar = args.tarfile

    # Extract
    with tarfile.open(tar) as f:
        f.extractall('./tmp_folder')

    # Read
    Q = np.load('./tmp_folder/qvals.npy')
    atoms = read('./tmp_folder/struc.traj')
    grid_info = np.loadtxt('./tmp_folder/plot.txt')

    # Delete tmp_folder
    shutil.rmtree('./tmp_folder')
    
    # Extract grid_anchor and scale
    grid_anchor = grid_info[:3]
    grid_scale = grid_info[-1]

    p = MainPlot(atoms,
                 Q,
                 grid_anchor = grid_anchor,
                 grid_scale = grid_scale)
    plt.show()
