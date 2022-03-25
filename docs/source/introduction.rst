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
    * Evaluator
    * Database

The **environment** defines the physical setup of the search problem including how many atoms and of which kind, 
the shape, the template, and the shape and size of the simulation cell. 

**Generators** are responsible for proposing new candidate structures, i.e. by randomly placing atoms the 
atoms that participate in the search in the simulation cell or in more guided ways. The generation 
procedure may be influenced by previously generated Candidates managed by a **sampler**. These candidates 
are collected by the **collector**. Candidates may be passed to a **postprocessor**, or a series of postprocessors, which 
e.g. may relax candidates in a model surrogate potential. An **acquisitor** can be used to determine the most promising 
candidate among a collection. Both acquisitors and the postprocessors may be influenced by a **model**, which could be a 
gaussian process regression model. The **evaluator** is responsible for calculating the properties of selected candidates, e.g.
the total energy in the true potential. Finally the database stores information about candidates. 

Each of these modules are implemented as classes with their own abstract base class that defines a number of required 
properties and methods that must be implemented for that type of module. So the AGOX script defines the interaction of 
these class instances/objects with each other. 

In general the class instances can be connected in two ways. Some classes use other classes and store instances of them as their 
own properties, e.g. the collector takes and stores instances of generators. The second way involves being observers of the 
main AGOX iterative loop. This is covered in the section about `My target`_.