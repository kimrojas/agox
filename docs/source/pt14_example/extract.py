from ase.io import write
from agox.modules.databases import Database

path = 'databases/db{}.db'

all_structures = []
for j in range(1, 11):
    database = Database(filename=path.format(j))
    structures = database.restore_to_trajectory()
    all_structures += structures

write('all_structures.traj', all_structures)


