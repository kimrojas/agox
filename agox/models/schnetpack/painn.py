import os
from pathlib import Path
import numpy as np

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import AtomsDataModule, ASEAtomsData

from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.calculators.calculator import Calculator, all_changes

from agox.models.ABC_model import ModelBaseClass
from agox.writer import agox_writer
from agox.observer import Observer

class PaiNN(ModelBaseClass):
    name = 'PaiNN-model'
    implemented_properties = ['energy', 'forces']

    """ SchNetPack PaiNN model

    Attributes
    ----------
    
    
    """
    
    def __init__(self, max_steps_per_iteration=100, max_epochs_per_iteration=10, cutoff=6., base_path='', db_name='dataset.db',
                 transfer_data=None, seed=None, **kwargs):
        super().__init__(**kwargs)

        self.reload_model = True
        
        if seed is not None:
            seed_everything(seed, workers=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.base_path = Path(base_path)

        VERSION_NUMBER = 0
        while True:
            self.version = Path(f'version_{VERSION_NUMBER}_seed_{seed}')
            self.train_path = self.base_path / self.version
            if not self.train_path.is_dir():
                self.train_path.mkdir(parents=True, exist_ok=True)
                break
            else:
                VERSION_NUMBER += 1

        self.db_name = Path(db_name)
        self.data_path = self.train_path / self.db_name
            
        self.cutoff = cutoff
        self.max_steps_per_iteration = max_steps_per_iteration
        self.max_epochs_per_iteration = max_epochs_per_iteration
        self.init_defaults(self.cutoff)


        # Training DB
        if self.data_path.is_file():
            self.writer('ASE Database already exist. \n Connecting to existing database.')
            self.spk_database = ASEAtomsData(datapath=str(self.data_path))
        else:
            self.spk_database = ASEAtomsData.create(
                str(self.data_path),
                distance_unit = 'Ang',
                property_unit_dict = {
                    'energy': 'eV',
                    'forces': 'eV/Ang'
                }
            )

        # Transfer data
        self.transfer_data = transfer_data
        if self.transfer_data is not None:        
            self.add_data(self.transfer_data)
        

        # Model
        representation = spk.representation.PaiNN(**self.defaults['representation'])
        pred_energy = spk.atomistic.Atomwise(n_in=self.defaults['representation']['n_atom_basis'],
                                             output_key='energy')
        pred_forces = spk.atomistic.Forces(energy_key='energy', force_key='forces')

        pairwise_distance = spk.atomistic.PairwiseDistances()        
        nnpot = spk.model.NeuralNetworkPotential(
            representation=representation,
            input_modules=[pairwise_distance],
            output_modules=[pred_energy, pred_forces],
            postprocessors=[
                trn.CastTo64(),
            ]
        )

        # Output
        output_energy = spk.task.ModelOutput(
            name='energy',
            loss_fn=torch.nn.MSELoss(),
            loss_weight=0.01,
            metrics={
                "MAE": torchmetrics.MeanAbsoluteError()
            }
        )

        output_forces = spk.task.ModelOutput(
            name='forces',
            loss_fn=torch.nn.MSELoss(),
            loss_weight=0.99,
            metrics={
                "MAE": torchmetrics.MeanAbsoluteError()
            }
        )

        self.task = spk.task.AtomisticTask(
            model=nnpot,
            outputs=[output_energy, output_forces],
            optimizer_cls=torch.optim.AdamW,
            optimizer_args={"lr": 1e-3},
            scheduler_cls=spk.train.ReduceLROnPlateau,
            scheduler_args={'factor': 0.5, 'patience': 1000, 'verbose': True},    
            scheduler_monitor = 'val_loss',
        )

        # Logging
        self.logger = pl.loggers.TensorBoardLogger(save_dir=str(self.base_path),
                                              name=None, version=str(self.version))
        self.callbacks = [
            spk.train.ModelCheckpoint(
                model_path=str(self.train_path / Path('best_inference_model')),
                monitor='val_loss',
                save_top_k=-1,
                save_last=True,
                every_n_epochs=1,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ]


    def init_defaults(self, cutoff):
        self.defaults = {
            'dataset': {
                'batch_size': 16,
                'num_train': 0.8,
                'num_val': 0.2,
                'transforms': [
                    trn.ASENeighborList(cutoff=cutoff),
                    trn.CastTo32()
                ],    
                'num_workers': 8,
                'pin_memory': True,
                'split_file': None #str(self.train_path / Path('split.npz')),
                
            },
            'representation': {
                'n_atom_basis': 96,
                'n_interactions': 5,
                'radial_basis': spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
                'cutoff_fn': spk.nn.CosineCutoff(cutoff)
            }
        }


    @property
    def transfer_data(self):
        return self._transfer_data
        
    @transfer_data.setter
    def transfer_data(self, l):
        if isinstance(l, list):
            self._transfer_data = l
            self._transfer_weights = np.ones(len(l))
        elif isinstance(l, dict):
            self._transfer_data = []
            self._transfer_weights = np.array([])
            for key, val in l.items():
                self._transfer_data += val
                self._transfer_weights = np.hstack((self._transfer_weights, float(key) * np.ones(len(val)) ))
        else:
            self._transfer_data = []
            self._trasfer_weights = np.array([])

    @property
    def transfer_weights(self):
        return self._transfer_weights

    
    def set_verbosity(self, verbose):
        self.verbose = verbose
    
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)        
        if 'energy' in properties:
            e = self.predict_energy(atoms=atoms)
            self.results['energy'] = e
        
        if 'forces' in properties:
            self.results['forces'] = self.predict_forces(atoms=atoms)

    ####################################################################################################################
    # Prediction
    ####################################################################################################################

    def predict_energy(self, atoms=None, X=None, return_uncertainty=False):
        if self.reload_model:
            self.load_model()
            
        a = atoms.copy()
        a.set_calculator(self.calculator)
        return a.get_potential_energy(apply_constraint=False)
        

    def predict_energies(self, atoms_list):
        if self.reload_model:
            self.load_model()
        return np.array([self.predict_energy(l) for l in atoms_list])
            

    def predict_uncertainty(self, atoms=None, X=None, k=None):
        self.writer('Uncertainty not implemented.')
        return 0.

        
    def predict_local_energy(self, atoms=None, X=None):
        self.writer('Local energy not implemented.')
        return np.zeros((len(atoms), ))


    def predict_forces(self, atoms, return_uncertainty=False, **kwargs):
        if self.reload_model:
            self.load_model()        
        a = atoms.copy()
        a.set_calculator(self.calculator)
        return a.get_forces(apply_constraint=False)


    @agox_writer
    def train_model(self, training_data, **kwargs):
        self.writer('Training PaiNN model')
        self.ready_state = True
        self.atoms = None

        self.add_data(training_data)
        
        # Dataloader
        dataset = AtomsDataModule(self.data_path, **self.defaults['dataset'])
        dataset.prepare_data()
        dataset.setup()


        trainer = pl.Trainer(
            accelerator=self.device.type,
            devices=1,
            callbacks=self.callbacks,
            logger=self.logger,
            default_root_dir=str(self.train_path),
            max_epochs=self.max_epochs_per_iteration,
            max_steps = self.max_steps_per_iteration,
            enable_progress_bar=False,
        )
        
        trainer.fit(self.task, datamodule=dataset)

        self.reload_model = True

            
    @agox_writer
    def update_model(self, new_data, all_data):
        self.add_data(new_data)
        self.train_model([])
                
                 
    ####################################################################################################################
    # Assignments:
    ####################################################################################################################

    @agox_writer
    @Observer.observer_method        
    def training_observer(self, database, state):
        iteration = state.get_iteration_counter()

        if iteration < self.iteration_start_training:
            return
        if (iteration % self.update_period != 0) * (iteration != self.iteration_start_training):
            return


        all_data = database.get_all_candidates()
        self.writer(f'lenght all data: {len(all_data)}')
        
        if self.ready_state:
            full_update = False
            data_amount_before = len(self.spk_database) - len(self.transfer_data)
            data_for_training = all_data
            data_amount_new = len(data_for_training) - data_amount_before
            new_data = data_for_training[-data_amount_new:] 
        else:
            full_update = True
            data_for_training = all_data

        if full_update:
            self.train_model(data_for_training)
        else:
            self.update_model(new_data, data_for_training)
        

    
    def load_model(self):
        self.writer('Loading model:', str(self.train_path / Path('best_inference_model')))
                    
        model = torch.load(str(self.train_path / Path('best_inference_model')),
                           map_location=self.device.type)
        
        self.calculator = spk.interfaces.SpkCalculator(
            model_file=str(self.train_path / Path('best_inference_model')),
            neighbor_list=trn.ASENeighborList(cutoff=self.cutoff),
            energy_key='energy',
            force_key='forces',
            energy_unit='eV',
            position_unit="Ang",
            device=self.device.type
        )
        self.reload_model = False

    def add_data(self, data_list):
        if len(data_list) == 0:
            return
        
        property_list = []
        for a in data_list:
            e = a.get_potential_energy(apply_constraint=False)
            f = a.get_forces(apply_constraint=False).reshape(-1, 3)
            c = SPC(a, energy=e, forces=f)
            a.set_calculator(c)
            properties = {'energy': np.array([e]), 'forces': f}
            property_list.append(properties)
        self.spk_database.add_systems(property_list, data_list)

