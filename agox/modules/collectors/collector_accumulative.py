from agox.modules.collectors.collector_ABC import CollectorBaseClass
import numpy as np
from ase.io import write
from ase import Atoms

from timeit import default_timer as dt

from agox.modules.helpers.plot_pca import plot_as_pca
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from agox.modules.candidates.candidate_standard import StandardCandidate
from agox.modules.models.gaussian_process.featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint

class AccumulativeCollector(CollectorBaseClass):

    name = 'AccumulativeCollector'

    def __init__(self, generators, environment, sampler, acquisitor, num_samples, verbose=False, plot_interval=100, pruning=0, energy_window=10, recalc_model_energy_batch_size=200, report_timing=False, wipe_period=100000, **kwargs):
        super().__init__(**kwargs)
        self.generators = generators
        self.num_samples = num_samples
        self.verbose = verbose
        self.plot_interval = plot_interval
        self.pruning = pruning # adding N new structures, N * pruning are removed
        self.report_timing = report_timing
        self.all_candidates_ever = []
        self.recalc_model_energy_index = 0
        self.recalc_model_energy_batch_size = recalc_model_energy_batch_size
        self.energy_window = energy_window
        self.number_of_new_candidates_added_last_episode = 0
        self.wipe_period = wipe_period

        self.sampler = sampler
        self.acquisitor = acquisitor
        self.envrionment = environment


    def make_candidates(self):
        
        # update the self.all_candidates_ever list (new model energies, remove the oldest candidates, etc)
        self.update_all_candidates_ever()

        # provide the acquisitor with candidates
        self.make_candidate_collection()

        # extend the self.all_candidates_ever list now that new candidates have been made
        self.extend_all_candidates_ever()

    def num_samples_this_episode(self):
        episode = self.get_episode_counter()
        print('NUM_SAMPLES IN EPISODE',episode,':', self.num_samples)
        return self.num_samples

    def make_candidate_collection(self):

        all_candidates = []

        t0 = dt()
        for generator, num_samples in zip(self.generators, self.num_samples_this_episode()):

            for sample in range(num_samples):
                candidates = generator(sampler=self.sampler, environment=self.envrionment)
                    
                for candidate in candidates:
                    all_candidates.append(candidate)
        
        # The generator may have returned None if it was unable to build a candidate.
        all_candidates = list(filter(None, all_candidates)) 

        if self.report_timing:
            print('Candidate generation time: {}'.format(dt()-t0))

        if self.verbose:
            write('candidate_ensemple_raw_{:05d}.traj'.format(self.get_episode_counter()),all_candidates)
        
        self.candidates = all_candidates
        print('Number of candidates this episode: {}'.format(len(self.candidates)))

    def update_all_candidates_ever(self):
        if self.get_episode_counter() % self.wipe_period == 0:
            print('COLL: WIPING')
            self.all_candidates_ever = []
            return

        # remove some of the oldest structures
        # if self.pruning = 0.3 and 15 new structures are about to be added, the 5 oldest structures will be removed
        print('PRUNING1',len(self.all_candidates_ever))
        N_pruning = int(self.pruning * self.number_of_new_candidates_added_last_episode)
        if 0 < N_pruning <= len(self.all_candidates_ever):
            self.all_candidates_ever = self.all_candidates_ever[N_pruning:]
        print('PRUNING2',len(self.all_candidates_ever))

        if len(self.all_candidates_ever) == 0:
            return

        if self.sampler.model_calculator.ready_state:
            recalc_from_this_index = self.recalc_model_energy_index
            if recalc_from_this_index > len(self.all_candidates_ever):
                recalc_from_this_index = 0
            recalc_to_this_index = recalc_from_this_index + min(self.recalc_model_energy_batch_size,len(self.all_candidates_ever))
            print('UPDATING from',recalc_from_this_index,'to',recalc_to_this_index,recalc_to_this_index % len(self.all_candidates_ever))
            for index in range(recalc_from_this_index,recalc_to_this_index):
                atoms = self.all_candidates_ever[index % len(self.all_candidates_ever)] 
                atoms.set_calculator(self.sampler.model_calculator)
                e = atoms.get_potential_energy()
                atoms.set_calculator(SPC(atoms, energy=e))
            self.recalc_model_energy_index = recalc_to_this_index % len(self.all_candidates_ever)

        if self.sampler.model_calculator.ready_state:
            print('CHOPPING1',len(self.all_candidates_ever))
            
            energies = [a.get_potential_energy() for a in self.all_candidates_ever]
            E0 = min(energies)
            self.all_candidates_ever = [self.all_candidates_ever[i] for i in np.arange(len(energies))[energies < E0 + self.energy_window]]
            
            print('CHOPPING2',len(self.all_candidates_ever))

    def extend_all_candidates_ever(self):
        new_candidates = []
        if len(self.candidates) > 0:
            temp_atoms = self.candidates[0]
            feature_calculator = Angular_Fingerprint(temp_atoms,Rc1=6,Rc2=4,binwidth1=0.2,Nbins2=30,
                                                     sigma1=0.2,sigma2=0.2,gamma=2,eta=20,use_angular=True)
            # Calculate features
            f_candidates = feature_calculator.get_featureMat(self.candidates)

            for candidate,f in zip(self.candidates,f_candidates):
    
                atoms = StandardCandidate(template=candidate.template, positions=candidate.positions, numbers=candidate.numbers, cell=candidate.cell)
    
                #atoms = Atoms(positions=candidate.positions, numbers=candidate.numbers, cell=candidate.cell)
                if self.sampler.model_calculator.ready_state:
                #if hasattr(self.sampler.model_calculator,'ever_trained') and self.sampler.model_calculator.ever_trained:
                    atoms.set_calculator(self.sampler.model_calculator)
                    e = atoms.get_potential_energy()
                else:
                    e = 0
                atoms.set_calculator(SPC(atoms, energy=e))
                atoms.add_meta_information('Standard Angular Fingerprint',f)
                new_candidates.append(atoms)

        if self.get_episode_counter() % self.plot_interval == 0:
            write('all_candidates_ever_{:06d}.traj'.format(self.get_episode_counter()),self.all_candidates_ever)
            plot_as_pca(self.all_candidates_ever,self.sampler.sample,new_candidates,self.get_episode_counter(),self.sampler.model_calculator)
     
        for candidate in new_candidates:
            self.all_candidates_ever.append(candidate)

        self.number_of_new_candidates_added_last_episode = len(new_candidates)

    def assign_from_main(self, main):
        super().assign_from_main(main)
        for generator in self.generators:
            generator.assign_from_main(main)
        


