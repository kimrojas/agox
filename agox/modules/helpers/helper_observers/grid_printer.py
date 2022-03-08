# This is ASLA-specific so perhaps it belongs somewhere else. 
# Put it here to have another easy to find example of how observers 
# can be used to follow a run. 

class GridPrinter:

    def __init__(self, memory, get_episode_counter, print_frequency=10):
        self.memory = memory
        self.get_episode_counter = get_episode_counter
        self.print_frequency = print_frequency
    
    def print_grid(self):
        if self.get_episode_counter() % self.print_frequency == 0:
            print(self.memory.grids[-1])

    def easy_attach(self, agox, order=10):
        agox.attach_observer('grid_printer', self.print_grid, order)

