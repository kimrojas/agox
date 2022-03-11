from __future__ import print_function
import numpy as np

from ase.calculators.calculator import Calculator
import os
import time
from random import randint
from ase.io import read, write, Trajectory


"""
Calculator for starting GPAW in parallel in a serial python job

call the calculator:
calc = SimpleGPAW(calculator='calc.py')
or:
calc = SimpleGPAW(calculator='calc.py',task='forces')

calc.py needs to defined:

########################cal.py file ###########################################
from ase.io import read
from ase.io import write, Trajectory
from ase.constraints import FixAtoms
from gpaw import *
import sys

#########################
##Do not delete. Important for reading geometry and writing output values
fname = sys.argv[1]
label = fname[:-5] + 'fin.traj'
t = read(fname)
#########################
calc = GPAW(mode='lcao',...)

t.set_calculator(calc)
t.get_potential_energy()
t.get_forces()
write(label,t)
########################end cal.py file #######################################
"""

class GPAW_IO(Calculator):
    """Simple GPAW calculator."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, calculator, parexe=True):
        self.calculator=calculator
        self.parexe=parexe
        self.n = 1

        self.read_path = 'structures_input.traj'
        self.write_path = 'structures_latest.traj'
        Calculator.__init__(self)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        #Trajectory(self.read_path, mode='w').write(atoms)

        # Set initial things:
        write(self.read_path, atoms)
        write(self.write_path, [])

        commandline = ''
        if self.parexe: 
            commandline += 'mpiexec '

        commandline += 'python ' + self.calculator + ' ' + self.read_path + ' ' + self.write_path
        
        # Execute the command-line:
        print(commandline)
        os.system(commandline)

        # Read the results:
        try:
            tnew = read(self.write_path)
        except Exception as e:
            self.error_handling(e)

        # Assign energies:
        self.results['energy'] = tnew.get_potential_energy()
        try: # Not sure when this can happen? 
            self.results['forces'] = tnew.get_forces()
        except Exception as e:
           self.error_handling(e)

        # Delete the files:
        os.remove(self.read_path)
        os.remove(self.write_path)
        

    def error_handling(self, exception):
         # Catch the error and print it
        print('Error: ', exception)
        # Delete files to avoid possibility of errors. 
        os.remove(self.read_path)
        os.remove(self.write_path)
        # Raise an exception that is caught by the gauge.
        # Unlike the error produced by GPAW this should only be printed once so much easier to search for.. 
        raise Exception('Energy calculation failed')