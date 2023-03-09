from ase import Atoms
from agox.models.descriptors import DescriptorBaseClass
from agox.models.descriptors.fingerprint_cython.angular_fingerprintFeature_cy import Angular_Fingerprint

class Fingerprint(DescriptorBaseClass):

    feature_types = ['global', 'global_gradient']

    name = 'Fingerprint'

    def __init__(self, rc1=6, rc2=4, binwidth=0.2, Nbins=30, sigma1=0.2, sigma2=0.2, gamma=2, 
                 eta=20, use_angular=True, *args, **kwargs):

        print('Initializing Fingerprint', args, kwargs)
        super().__init__(*args, **kwargs)

        init_atoms = self.environment.get_template()
        init_atoms += Atoms(self.environment.get_numbers())
        
        self.cython_module = Angular_Fingerprint(init_atoms, Rc1=rc1, Rc2=rc2, 
            binwidth1=binwidth, Nbins2=Nbins, sigma1=sigma1, sigma2=sigma2, 
            gamma=gamma, eta=eta, use_angular=use_angular)

    def create_global_features(self, atoms):
        return self.cython_module.get_feature(atoms)

    def create_global_feature_gradient(self, atoms):
        return self.cython_module.get_featureGradient(atoms)

    @classmethod
    def from_atoms(cls, atoms, **kwargs):
        from agox.environments import Environment
        environment = Environment(template=atoms, symbols='', use_box_constraint=False,
                                  print_report=False)
        return cls(environment=environment, **kwargs)

        
