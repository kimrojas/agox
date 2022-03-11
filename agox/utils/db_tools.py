import numpy as np
import sqlite3
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write,read
from ase.atoms import Atoms,Atom
import json
import os
import inspect

def get_row_num(names,name):
    """
    Returns the row number where names==name
    """
    return int(np.where(names==name)[0])

def _rebuild_struc_from_row(structure_row,grid_row,names):
    """
    """
    # Extract information
    positions = np.frombuffer(structure_row[2],float).reshape(-1,3)
    atom_types = np.frombuffer(structure_row[4],float).astype(int)
    E = structure_row[3]
    cell = np.frombuffer(structure_row[5],float).reshape(3,3)
    # Build structure
    atom_list = []
    for p,a in zip(positions,atom_types):
        atom_list.append(Atom(symbol=a,position=p))
    atoms = Atoms(atom_list,cell=cell,pbc=[True,True,True])
        
    # Attach the energy to the grid
    calcSingle = SinglePointCalculator(atoms, energy=E)
    atoms.set_calculator(calcSingle)
    
    return atoms

def _rebuild_struc_from_cur(cur,i):
    """
    Rebuilds a structure with id = i from a cursor of a database object
    """
    cur.execute("SELECT * FROM structures WHERE id=?", (int(i),))
    structure_row = cur.fetchall()[0]

    if structure_row is None:
        return None

    cur.execute("SELECT * FROM structures")
    names = np.array([d[0] for d in cur.description])
    grid_row = cur.fetchone()

    g = _rebuild_struc_from_row(structure_row,grid_row,names)
    return g

def _rebuild_struc_from_conn(conn,i):
    """
    Rebuilds a structure with id = i from a connection a to database object
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM structures WHERE id=?", (int(i),))

    structure_row = cur.fetchall()[0]

    if structure_row is None:
        return None

    cur.execute("SELECT * FROM structures")
    names = np.array([d[0] for d in cur.description])
    grid_row = cur.fetchone()

    g = _rebuild_struc_from_row(structure_row,grid_row,names)

    return g


def rebuild_from_db(database, idx = 1):
    """
    Rebuilds the idx'th structure from the database and returns the Atoms object
    """

    if type(database) is str:
        conn = sqlite3.connect(database)
    else:
        conn = database.con

    if type(idx) is list:
        list_of_structures = True
        idx_list = idx
    else:
        list_of_structures = False
        idx_list = [idx]        


    grids = []
    for i in idx_list:
        g = _rebuild_struc_from_conn(conn,i)
        # Get the structure information from the database
        if g is None:
            if type(database) is str:
                print('No structure with index %i in %s' %(i,database))
            else:
                print('No structure with index %i in %s' %(i,database.filename))
        grids.append(g)

    if list_of_structures:
        return grids
    else:
        return grids[0]



def get_energies(database):
    """
    Returns the energies of the structures in the database
    """
    if type(database) is str:
        conn = sqlite3.connect(database)
        cur = conn.cursor()
    else:
        conn = database.con
        cur = conn.cursor()

    cur.execute("SELECT energy FROM structures")
    energies = np.array(cur.fetchall()).flatten()
    return energies

   
    
def get_all(database,N=None,flush=False,verbose = False):
    """
    Returns all the structures in the database. If N is not None, the function returns the first N structures
    """
    
    if type(database) is str:
        conn = sqlite3.connect(database)
        cur = conn.cursor()
    else:
        conn = database.con
        cur = conn.cursor()


    if N is None:
        cur.execute("SELECT id FROM structures")
        idxes = cur.fetchall()
        idxes = [i[0] for i in idxes]
    else:
        idxes = range(1,N+1)
    structs = []
    for i in idxes:
        structs.append(_rebuild_struc_from_cur(cur,i))
        # print('Getting structure: %i/%i' %(i,len(idxes)),end='\r', flush=flush)
        if verbose:
            if i%50==0:
                print('Getting structure: %i/%i' %(i,len(idxes)), flush=flush)
    if verbose:                
        print('')
    return structs

def get_run():
    """
    Collects all trajs in all db files in the working directory
    """

    db_list = []
    for f in os.listdir():
        if '.db' in f:
            db_list.append(f)


    trajs = None
    
    for db in db_list:
        all_from_db = get_all(db)
        if trajs is None:
            trajs = all_from_db
        else:
            for t in all_from_db:
                trajs.append(t)
                

        # all_trajs.append(trajs)


    write('all_trajs.traj',trajs)

        
def get_some(database,idx_list,flush=False,verbose = False):
    """
    """

    if type(database) is str:
        conn = sqlite3.connect(database)
        cur = conn.cursor()
    else:
        conn = database.con
        cur = conn.cursor()

    structs = []
    
    for m,i in enumerate(idx_list):
        if verbose:
            print('Getting structure: %i/%i' %(m,len(idx_list)),end='\r', flush=flush)
        structs.append(_rebuild_struc_from_conn(conn,i+1)) # +1 because the id in the database starts from 1
    return structs

def get_best(database,N=1,up_to_episode=None,return_episode=False,flush = False,verbose = False):
    """
    Returns the N best structures in database
    """
    if verbose:
        print('Getting structure from database %s' %(database),end='\r', flush=flush)
        print('')
    if not os.path.exists(database):
        raise IOerror('Database %s does not exist' %(database))

    if type(database) is str:
        conn = sqlite3.connect(database)
        cur = conn.cursor()
    else:
        conn = database.con
        cur = conn.cursor()

    cur.execute("SELECT energy FROM structures")
    energy = np.array(cur.fetchall()).flatten()
    if up_to_episode is not None:
        energy=energy[:up_to_episode]


    # check for 'None' energies
    energy_to_sort = energy.copy()
    none_idx = np.array([e is None for e in energy_to_sort])
    not_none_idx = np.array([e is not None for e in energy_to_sort])
    energy_to_sort[none_idx] = np.max(energy_to_sort[not_none_idx])+1
    args = np.argsort(energy_to_sort)
    total_num_episodes = len(energy)
    
    structs = []
    for i in args[:N]:
        structs.append(_rebuild_struc_from_conn(conn,i+1)) # +1 because the id in the database starts from 1

    if return_episode:
        if N==1:
            return structs[0],args[0]+1,total_num_episodes
        else:
            return structs,args[:N]+1,total_num_episodes
    else:
        if N==1:
            return structs[0]
        else:
            return structs


def get_worst(database,N=1,return_episode=False,flush = False):
    """
    Returns the N worst structures in database
    """
    print('Getting structure from database %s' %(database),end='\r', flush=flush)
    print('')
    if not os.path.exists(database):
        print('Database %s does not exist' %(database))
        return None,None,None
        
        
    if type(database) is str:
        conn = sqlite3.connect(database)
        cur = conn.cursor()
    else:
        conn = database.con
        cur = conn.cursor()

    cur.execute("SELECT energy FROM structures")
    energy = np.array(cur.fetchall()).flatten()
    args = np.argsort(energy)[::-1]
    total_num_episodes = len(energy)
    
    structs = []
    for i in args[:N]:
        structs.append(_rebuild_struc_from_conn(conn,i+1)) # +1 because the id in the database starts from 1

    if return_episode:
        if N==1:
            return structs[0],args[0]+1,total_num_episodes
        else:
            return structs,args[:N]+1,total_num_episodes
    else:
        if N==1:
            return structs[0]
        else:
            return structs
