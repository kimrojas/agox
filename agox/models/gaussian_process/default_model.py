from ase import Atoms
from agox.models.gaussian_process.featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from agox.models.gaussian_process.delta_functions_multi.delta import delta as deltaFunc
from agox.models.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from agox.models.gaussian_process.GPR import GPR

def get_default_GPR_model(environment=None, feature_calculator=None, atoms=None, grid=None, builder=None, use_delta_func=True, use_delta_in_training=False,
                          beta=0.01, theta0=None,theta0ini = 1**2, theta0min = 1e-6, theta0max = 1e3,
                          lambda1 = None, lambda1ini = 1.8, lambda1min = 0.1, lambda1max = 100,
                          lambda2 = None, lambda2ini = 0.35,lambda2min = 0.1, lambda2max = 10, return_feature_calc=False,
                          sigma_noise = None,
                          use_correct_kernel = False):

    assert use_delta_in_training is False; 'Using delta in training is NOT recommended'

    needs_temp_atoms = (feature_calculator is None) or use_delta_func
    if needs_temp_atoms:
        if grid is not None:
            temp_atoms = grid.copy()
            #for species in builder.numbers:
            temp_atoms += Atoms(builder.numbers)
        elif atoms is not None:
            temp_atoms = atoms.copy()
        elif environment is not None:
            temp_atoms = environment.get_template()
            temp_atoms += Atoms(environment.get_numbers())
        else:
            print('Need eiter atoms object or grid/builder as input arguments.')
    
    if feature_calculator is None:
        Rc1 = 6
        binwidth1 = 0.2
        sigma1 = 0.2
    
        # Angular part
        Rc2 = 4
        Nbins2 = 30
        sigma2 = 0.2
        gamma = 2
    
        # Radial/angular weighting
        eta = 20 
        use_angular = True
    
        # Initialize feature
        feature_calculator = Angular_Fingerprint(temp_atoms,
                                                Rc1=Rc1,
                                                Rc2=Rc2,
                                                binwidth1=binwidth1,
                                                Nbins2=Nbins2,
                                                sigma1=sigma1,
                                                sigma2=sigma2,
                                                gamma=gamma,
                                                eta=eta,
                                                use_angular=use_angular)
    
    if theta0 is not None:
        theta0ini = theta0
        theta0min = theta0
        theta0max = theta0
    if lambda1 is not None:
        lambda1ini = lambda1
        lambda1min = lambda1
        lambda1max = lambda1
    if lambda2 is not None:
        lambda2ini = lambda2
        lambda2min = lambda2
        lambda2max = lambda2

    if use_correct_kernel:
        if sigma_noise is None:
            sigma_noise = 1e-2
        kernel = C(theta0ini, (theta0min, theta0max)) * \
             ( \
               C((1-beta), ((1-beta), (1-beta))) * RBF(lambda1ini, (lambda1min,lambda1max)) + \
               C(beta, (beta, beta)) * RBF(lambda2ini, (lambda2min,lambda2max)) 
             ) + \
             WhiteKernel(sigma_noise, (sigma_noise,sigma_noise))
    else:
        if sigma_noise is None:
            sigma_noise = 1e-5
        kernel = C(theta0ini, (theta0min, theta0max)) * \
             ( \
               C((1-beta), ((1-beta), (1-beta))) * RBF(lambda1ini, (lambda1min,lambda1max)) + \
               C(beta, (beta, beta)) * RBF(lambda2ini, (lambda2min,lambda2max))  + \
               WhiteKernel(sigma_noise, (sigma_noise,sigma_noise))
             )

    if use_delta_func:
        delta = deltaFunc(atoms=temp_atoms, rcut=6)
    else:
        delta = None

    gpr = GPR(kernel=kernel,
            featureCalculator=feature_calculator,
            delta_function=delta,
            # delta_function=None,          
            bias_func=None,
            optimize=True,
            n_restarts_optimizer=1,
            use_delta_in_training=use_delta_in_training)

    if not return_feature_calc:
        return gpr
    else:
        return gpr, feature_calculator
