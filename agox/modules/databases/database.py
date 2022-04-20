import numpy as np 
from time import time
from ase import Atoms
import sqlite3
import os
import sys
import os.path

from .database_utilities import *
from .database_ABC import DatabaseBaseClass
from copy import deepcopy
from collections import OrderedDict


init_statements = ["""create table structures (
id integer primary key autoincrement,
ctime real,
positions blob,
energy real,
type blob,
cell blob,
forces blob, 
pbc blob,
template_indices blob
)"""]

# Should match init_statements!!
unpack_functions = [nothing, nothing, deblob, nothing, deblob, deblob, deblob, deblob, deblob]
pack_functions = [nothing, nothing, blob, nothing, blob, blob, blob, blob]

from agox.modules.candidates import StandardCandidate

class Database(DatabaseBaseClass):
    """ Database module """

    name = 'Database'

    def __init__(self, filename='db.db', initialize=False, verbose=False, write_frequency=50, **kwargs):
        super().__init__(**kwargs)

        # General stuff:
        self.verbose = verbose

        # File-based stuff:
        self.filename = filename
        if initialize and os.path.exists(filename):
            os.remove(filename)
        self.con = sqlite3.connect(filename, timeout=600)
        self.write_frequency = write_frequency

        # Important that this matches the init_statements list. 
        self.storage_keys = ['positions', 'energy', 'type', 'cell', 'forces', 'pbc', 'template_indices']

        # Memory-based stuff:        
        self.candidate_instantiator = StandardCandidate
        self.candidate_energies = []

        self._initialize()
    ####################################################################################################################
    # Memory-based methods:
    ####################################################################################################################

    def store_candidate(self, candidate, accepted=True, write=True, dispatch=True):
        # Needs some way of handling a dummy candidate, probably boolean argument.
        if accepted:
            self.candidates.append(candidate)           
            self.candidate_energies.append(candidate.get_potential_energy())
        if write:
            self.write(candidate)
        if dispatch:
            self.dispatch_to_observers(self)

    def get_all_candidates(self):
        all_candidates = []
        for candidate in self.candidates:
            all_candidates.append(candidate)
        return all_candidates

    def get_most_recent_candidate(self):
        if len(self.candidates) > 0:
            candidate = deepcopy(self.candidates[-1]) # Not entirely sure why this deepcopy is neccesary, but .copy does not work. / Something with the calculator not being copieda
        else:
            candidate = None
        return candidate

    def get_recent_candidates(self, number):
        return [deepcopy(candidate) for candidate in self.candidates[-number:]]

    def get_best_energy(self):
        return np.min(self.candidate_energies)
        
    ####################################################################################################################
    # File-based methods:
    ####################################################################################################################

    def _init_storage(self):
        self.positions = []
        self.atom_numbers = []
        self.energies = []
        self.cells = []
        self.forces = []
        self.pbc = []
        self.template_indices = []

        self.storage_dict = OrderedDict()
        for key in self.storage_keys:
            self.storage_dict[key] = []
        
        self.number_of_rows = 0
        
    def _initialize(self):
        # Check if structures are in the database, otherwise initialize tables
        cur = self.con.execute(
            'select count(*) from sqlite_master where name="structures"')
        if cur.fetchone()[0] == 0:
            for statement in init_statements:
                self.con.execute(statement)
            self.con.commit()
        self._init_storage()
        self.con.row_factory = sqlite3.Row

    def store_information(self, candidate):
        if candidate is not None:
            e = candidate.calc.results.get('energy', 0)
            f = candidate.calc.results.get('forces', np.zeros(candidate.positions.shape))
            
            self.storage_dict['energy'].append(e)
            self.storage_dict['forces'].append(f)
            self.storage_dict['positions'].append(np.array(candidate.positions, dtype=np.float64))
            self.storage_dict['type'].append(np.array(candidate.numbers, dtype=np.float64))
            self.storage_dict['cell'].append(np.array(candidate.cell, dtype=np.float64))
            self.storage_dict['pbc'].append(np.array(candidate.pbc.astype(int), dtype=np.float64))
            self.storage_dict['template_indices'].append(np.array(candidate.get_template_indices(), dtype=np.float64))
            self.number_of_rows += 1

    def get_row_to_write(self, index):
        row = [now()]
        for (key, value), func in zip(self.storage_dict.items(), pack_functions):
            row.append(func(value[index]))
        return tuple(row)

    def write(self, candidate=None, force_write=False):

        self.store_information(candidate)

        if self.number_of_rows == self.write_frequency or force_write:
            for row_index in range(self.number_of_rows):
                cur = self.con.cursor()
                row = self.get_row_to_write(row_index)
                q = 'NULL,' + ', '.join('?' * len(row))

                cur.execute('insert into structures values ({})'.format(q),
                            row)

                self.con.commit()
            self._init_storage()          

    def db_to_atoms(self, structure):
        """

        Converts a database representation (dictionary) of a structure to an ASE atoms object

        Parameters
        ----------
        structure :  database representation of a structure

        Returns
        -------
        struc : ASE Atoms object

        """
        e = structure['energy']
        f = structure.get('forces', 0)
        pos = structure['positions']
        num = structure['type']
        cell = structure['cell']
        pbc = structure.get('pbc', None)

        atoms = Atoms(symbols = num,
                    positions = pos,
                    cell = cell, 
                    pbc=pbc)        

        calc = SinglePointCalculator(atoms, energy=e, forces=f)
        atoms.set_calculator(calc)
        return atoms

    def db_to_candidate(self, structure):
        e = structure['energy']
        f = structure.get('forces', 0)
        pos = structure['positions']
        num = structure['type']
        cell = structure['cell']
        pbc = structure.get('pbc', None)
        template_indices = structure.get('template_indices', None)

        if hasattr(self, 'template') and template_indices is None:
            template = self.template
        else:
            template = None            

        candidate = self.candidate_instantiator(symbols = num, positions = pos, cell = cell, pbc=pbc, template=template, template_indices=template_indices)        
        calc = SinglePointCalculator(candidate, energy=e, forces=f)
        candidate.set_calculator(calc)
        return candidate

    def get_data_from_row(self, row):
        d = {}
        for key, value, func in zip(row.keys(), row, unpack_functions):
            d[key] = func(value)

        d['positions'] = d['positions'].reshape(-1, 3)
        d['cell'] = d['cell'].reshape(3, 3)

        return d

    def get_all_structures_data(self):
        cursor = self.con.execute("SELECT * from structures")
        structures = []
        for row in cursor.fetchall():
            d = self.get_data_from_row(row)
            structures.append(d)
        return structures

    def get_structure_data(self, i):
        t = (str(int(i)),)
        cursor = self.con.execute('SELECT * from structures WHERE id=?',t)
        row = cursor.fetchone()
        d = self.get_data_from_row(row)    
        return d        

    def get_all_energies(self):
        cursor = self.con.execute("SELECT energy from structures")
        energies = []
        for row in cursor.fetchall():
            energies.append(row['energy'])
        return np.array(energies)

    def get_best_structure(self):
        structure_data = self.get_all_structures_data()
        energies = [c['energy'] for c in structure_data]
        idx = energies.index(min(energies))
        best_struc = self.db_to_atoms(structure_data[idx])        
        return best_struc
        
    def get_structure(self, index):
        cand = self.get_structure_data(index)
        struc = self.db_to_atoms(cand)

        return struc

    def restore_to_memory(self):
        strucs = self.get_all_structures_data()
        candidates = []
        for struc in strucs:
            candidates.append(self.db_to_candidate(struc))
        self.candidates = candidates

    def restore_to_trajectory(self):
        strucs = self.get_all_structures_data()
        atoms_objects = []
        for structure in strucs:
            atoms_objects.append(self.db_to_atoms(structure))
        return atoms_objects

    ####################################################################################################################
    # Misc methods:
    ####################################################################################################################

    def assign_from_main(self, main):
        super().assign_from_main(main)
        main.attach_finalization('Database write', lambda: self.write(candidate=None, force_write=True))
