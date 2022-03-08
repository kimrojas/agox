Abstract Base Classes
=====================

In AGOX all of the main elements have an abstract base class that details that must be inherited from when implementing
a specific version. This is to ensure compatibility and consistent naming of methods that may be used by other modules. 

As an example we can look at the generator base class 

.. literalinclude:: ../../agox/modules/generators/generator_ABC.py
    :language: python

We see that the base class imposes that the 'get_candidates'-method must be defined, that it must take as arguments 
a sampler and an environment and that we must define the attribute 'self.name' of any class that inherits from this one. 
The remaining methods are methods that may be convenient for a wide range of generators, such as checking bond lengths 
or generating vectors. There are also methods for checking that the solutions obey the confinement cell. 

Looking at the base-class and understanding which methods have the @abstractmethod decorator should always be the first 
step when implementing a version of that element. Furthermore you should not rely on methods that are not stated (abstract or default) in the 
ABC when interacting an instance of a class in another module, as other functions are not guaranteed to exists for all versions of that 
class so doing so will potentially cause crashes. Note, this does not mean that you should update the ABC with new methods unless 
you have very good reason to do so, changes to the ABC's are the least likely to be accepted!