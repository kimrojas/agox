import pickle
import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class GaussianProcessCalculator(Calculator):
    """
    Model calculator that trains on structures form ASLA memory.
    """

    implemented_properties = ['energy', 'forces']
    
    def __init__(self, model=None, **kwargs):
        self.results = {}
        self.model = model
        
        super().__init__(**kwargs)
        
    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.atoms = atoms.copy()
        self.atoms.pbc = [1,1,1]

        #E,E_err,_ = self.model.predict_energy(self.atoms, return_error=True)
        #self.results['energy'] = E - self.kappa*E_err
        self.results['energy'] = self.model.predict_energy(self.atoms, return_error=False)
        
        if 'forces' in properties:
            forces,f_err = self.model.predict_force(self.atoms,return_error = True)
            F = (forces+self.kappa*f_err).reshape(len(self.atoms),3)
            for c in self.atoms.constraints:
                c.adjust_forces(self.atoms,F)
            self.results['forces'] = F

    def train(self, trajectory):
        """
        Train GP model on the given trajectory file.
        """
        self.model.train(trajectory, optimize=True)
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
                



if __name__ == '__main__':

    from ase.io import read
    from agox.modules.gaussian_process.default_model import get_default_GPR_model
    import matplotlib.pyplot as plt

    for energy_offset in [10, 20, 30, 40, 50]:
        traj_path = '/home/machri/data_from_bjork/combined.traj'
        traj = read(traj_path, index=':')

        # Filter a little bit:
        all_energies = np.array([atoms.get_potential_energy() for atoms in traj])
        index = np.max(np.argwhere(all_energies < all_energies[0] + energy_offset))
        traj = traj[0:index]
        

        # Make and train
        model = get_default_GPR_model(traj[0])
        gp_calc = GaussianProcessCalculator(model=model)
        gp_calc.train(traj)
        gp_calc.save_model('test_models/model_{}.pckl'.format(energy_offset))

        k1 = gp_calc.model.kernel_.get_params()['k1']
        l1 = gp_calc.model.kernel_.get_params()['k2__k1__k1__k2__length_scale']
        l2 = gp_calc.model.kernel_.get_params()['k2__k1__k2__k2__length_scale']

        print('L1: {:<20} L2: {:<20}K1: {:<20}'.format(l1,l2,str(k1)))

        # Load 
        # gp_calc = GaussianProcessCalculator()
        # gp_calc.load_model('model.pckl')

        # Test
        true_energies = np.array([atoms.get_potential_energy() for atoms in traj])
        energies = np.zeros(len(traj))
        for i, atoms in enumerate(traj):
            if i % 25 == 0:
                print(i)
            atoms.set_calculator(gp_calc)
            energies[i] = atoms.get_potential_energy()
            
        # Plot
        error = np.abs(energies-true_energies)
        np.save('test_models/error_{}.npy'.format(energy_offset), error)
        # fig, ax = plt.subplots()
        # ax.set_title('Mean error = {}'.format(mean_error))
        # ax.plot(np.abs(energies-true_energies))
        # plt.show()

        



        




