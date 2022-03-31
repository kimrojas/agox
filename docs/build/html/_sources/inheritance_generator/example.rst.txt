Inheritance: An example
=======================

As an example of the OOP framework that AGOX employs a closer look at one of the elements is helpful, specifically the 
generator. As with all modules all generators are based on an ABC that implements a number of methods. Generally the global 
search problem is a configurational problem, that is the difficulty comes from determining bonding patterns which we do not 
wish to encode preconceived ideas about, however for bond lengths an appropriate range can be determined from the covalent radii 
of the involved atoms. Therefore the generator ABC implements a common method to check that all atoms are within such a range, e.g 
from 3/4 to 5/4 of the sum of covalent radii. Additionally the ABC implements methods for checking that atoms are within a 
specified cell, and various methods for generating positions. The ABC has one abstract method 'get_candidates'
which is what must be implemented for any specific generator, which we would want by looking at the base-class:

.. literalinclude:: ../../../agox/modules/generators/generator_ABC.py
    :language: python
    :lines: 11-50

Where the decorator @abstrathmethod tells us that this is a method that has to be implemeneted. 

One such specific generator is what we call a 'Replace Generator' 
where the 'get_candidates' method is implemented as such:

.. literalinclude:: ../../../agox/modules/generators/replace_generator.py
    :language: python
    :lines: 8-41

The behaviour of the generator is controlled by four methods that are invoked by the 'get_candidates method', 
namely 'get_indices_to_move', 'get_radius', 'get_displacement_vector' and 'get_new_position_center'. 
Where 'get_displacement_vector' is default method of the generator ABC. The intended behaviour of the generator is 
also very clear from these four functions, first a set of atoms to be moved are chosen, then they are moved one after 
the other by choosing a radius, getting a vector of that radius and choosing the center around where that displacement 
is made.  

% Note OOP still requires good coding practices, this method would not be nearly as versatile if it had not been using 
% sub-methods calls!

In the straightforward implementation of the 'Replace Generator' both 'get_indices_to_move' and 'get_radius'
are implemented to choose the properties they control in a uniformly random way and 'get_new_position_center'
always chooses the current position of the atom in question. We can easily implement a version of this generator, that is 
beneficial for surface clusters, by inherting from the ReplaceGenerator but changing these few methods, like so: 

.. literalinclude:: ../../../agox/modules/generators/cog_generator.py
    :language: python

What should be noted here is that code written in this is very adaptable due to the usage of inheritance. Furthermore, 
experiments such as the one documented here posses no risk of introducing bugs in established parts of the code-base, as 
the interaction with established methods happens through inheritance.