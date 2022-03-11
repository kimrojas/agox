Introduction to AGOX 
====================

The behaviour of an AGOX search is controlled through several modules, with a search generally requiring 
the choice of 
    * Environment
    * Generators
    * Collector
    * Postprocessors
    * Acquisitor
    * Model
    * Sampler
    * Database

The **environment** defines the physical setup of the search problem including how many atoms and of which kind, 
the shape, the template, and the shape and size of the simulation cell. 

**Generators** are responsible for proposing new candidate structures, i.e. by randomly placing atoms the 
atoms that participate in the search in the simulation cell or in more guided ways. The generation 
procedure may be influenced by previously generated Candidates managed by a **sampler**. These candidates 
are collected by the **collector** which can apply **postprocessors** such as relaxation in a surrogate 
model to each proposed candidate. Before the **acquisitor** chooses a candidate for which to do an 
energy calculation. This may then be used to update any **models** use by e.g. the acquisitor or the postprocessors. 
The results are stored in a **database**. 

.. figure:: images/agox_flowchart_paper.pdf