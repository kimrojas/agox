Ru3N4C4 (DFT GPAW): Runscript
==========================

.. literalinclude:: Ru3N4C4_k111.py 
    :language: python
    :linenos:

Most of the script is pretty much the same as the previous Pt14 example, except for the slight changes required to use a 
DFT calculator: 

.. literalinclude:: Ru3N4C4_k111.py
    :language: python
    :lines: 38-48

This calculator takes the same arguments as a normal GPAW calculator, however with the important exception that ALL 
arguments are given as strings, as an example:

.. code-block:: python

    mode='PW(300)'

This will be interpreted as the GPAW PW-function. If you're using more exotic settings that require require functions from the GPAW library 
then they can be loaded with the modules argument 

.. code-block:: python

     modules=['from gpaw.utilities import h2gpts', 'from gpaw import FermiDirac'])

This loads the 'h2gpts' and 'FermiDirac' functions from GPAW. 

The calculator spawns another process that runs in parallel using MPI, you can set the shell command used to invoke MPI 
with the 'mpi_command' argument: 

.. code-block:: python

     calc = GPAW_IO(..., mpi_command='mpiexec')

Which may be changed to whatever is appropriate on your system, e.g. 'srun' on Slurm clusters.

Note that the calculator uses IO operations (ase.read/write) to communicate with the secondary process which adds a slight overhead, however 
it is rather minor unless the DFT calculation is very fast!
