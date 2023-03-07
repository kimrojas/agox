import os
from pathlib import Path

import io
from contextlib import redirect_stdout, redirect_stderr

import numpy as np

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import AtomsDataModule, ASEAtomsData
from schnetpack.representation import PaiNN
from schnetpack.interfaces import AtomsConverter

from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.calculators.calculator import Calculator, all_changes

from agox.models.ABC_model import ModelBaseClass
from agox.writer import agox_writer
from agox.observer import Observer
from ase import Atoms

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class SchNetPackModel(ModelBaseClass):
    name = 'SchNetPack-model'

    implemented_properties = ['energy', 'forces']

    dynamic_attributes = ['nnpot']

    """ SchNetPack PaiNN model

    Attributes
    ----------
    
    
    """

    def __init__(
            self,
            cutoff=6.,
            dataset_settings={},
            representation_settings={},
            loss_settings={},
            learning_settings={},
            trainer_callbacks=[],
            max_steps_per_iteration=100,
            max_epochs_per_iteration=10,
            representation_cls=PaiNN,
            base_path='',
            training_device=None,
            prediction_device=None,
            db_name='dataset.db',
            transfer_data=None,
            tensorboard=True,
            seed=None,
            **kwargs
    ):
        
        super().__init__(**kwargs)

        self.tensorboard = tensorboard
        if seed is not None:
            seed_everything(seed, workers=True)

        if training_device is None:
            self.training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.training_device = torch.device(training_device)

        if prediction_device is None:
            self.prediction_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.prediction_device = torch.device(prediction_device)

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

        self.trainer_iteration = 1
            
        self.cutoff = cutoff
        self.dataset_settings = {**dataset_settings,
                                 **{
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
                                 }
        }
        self.representation_settings = {**representation_settings,
                                        **{
                                            'n_atom_basis': 96,
                                            'n_interactions': 5,
                                            'radial_basis': spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
                                            'cutoff_fn': spk.nn.CosineCutoff(cutoff)
                                        }
        }
        self.loss_settings = {**loss_settings,
                              **{
                                  'energy_weight': 0.01,
                                  'forces_weight': 0.99
                              }
        }
        self.learning_settings = {**learning_settings,
                                  **{
                                      'optimizer_cls': torch.optim.AdamW,
                                      'optimizer_args': {"lr": 1e-4},
                                      'scheduler_cls': spk.train.ReduceLROnPlateau,
                                      'scheduler_args': {'factor': 0.5, 'patience': 1000, 'verbose': True},
                                      'scheduler_monitor': 'val_loss',
                                  }
        }

        self.ckpt_path = self.train_path / 'last.ckpt'
        self.callbacks = trainer_callbacks + [pl.callbacks.ModelCheckpoint(dirpath=str(self.train_path), filename='last', save_last=True)]
        # self.callbacks = [
        #     spk.train.ModelCheckpoint(
        #         model_path=str(self.train_path / Path('best_inference_model')),
        #         monitor='val_loss',
        #         save_top_k=-1,
        #         save_last=True,
        #         every_n_epochs=1,
        #     ),
        #     pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        # ]
        
        self.max_steps_per_iteration = max_steps_per_iteration
        self.max_epochs_per_iteration = max_epochs_per_iteration
        self.representation_cls = representation_cls

        
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
        representation = self.representation_cls(**self.representation_settings)
        pred_energy = spk.atomistic.Atomwise(n_in=self.representation_settings.get('n_atom_basis'),
                                             output_key='energy')
        pred_forces = spk.atomistic.Forces(energy_key='energy', force_key='forces')

        pairwise_distance = spk.atomistic.PairwiseDistances()
        
        self.nnpot = spk.model.NeuralNetworkPotential(
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
            loss_weight=self.loss_settings.get('energy_weight'),
            metrics={
                "MAE": torchmetrics.MeanAbsoluteError()
            }
        )

        output_forces = spk.task.ModelOutput(
            name='forces',
            loss_fn=torch.nn.MSELoss(),
            loss_weight=self.loss_settings.get('forces_weight'),
            metrics={
                "MAE": torchmetrics.MeanAbsoluteError()
            }
        )

        self.task = spk.task.AtomisticTask(
            model=self.nnpot,
            outputs=[output_energy, output_forces],
            **self.learning_settings,
        )

        # Logging
        if self.tensorboard:
            self.logger = pl.loggers.TensorBoardLogger(save_dir=str(self.base_path),
                                                       name=None, version=str(self.version))
        else:
            self.logger = False


        ########## FOR PREDICTION ##########
        self.converter = AtomsConverter(
            neighbor_list=trn.ASENeighborList(cutoff=self.cutoff),
            device=self.prediction_device.type,
            dtype=torch.float32,
            
        )
            
        self.energy_key = 'energy'
        self.forces_key = 'forces'

        
            

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
    
    
    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        self.nnpot.eval()
        self.nnpot.to(device=self.prediction_device.type)#, dtype=torch.float64)

        # Something inside schnetpack is checking for atoms objects so converting here. 
        atoms = Atoms(numbers=atoms.get_atomic_numbers(), positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        
        model_inputs = self.converter(atoms)
        model_results = self.nnpot(model_inputs)
        if 'energy' in properties:
            e = model_results['energy'].cpu().data.numpy()[0]
            # print('Energy pred:', type(e), e, flush=True)
            self.results['energy'] = e
        
        if 'forces' in properties:
            f = model_results['forces'].cpu().data.numpy()
            # print('Forces pred:', type(f), f.shape, flush=True)
            self.results['forces'] = f


    ####################################################################################################################
    # Prediction
    ####################################################################################################################

    def predict_energy(self, atoms=None, X=None, return_uncertainty=False):
        self.nnpot.eval()
        self.nnpot.to(device=self.prediction_device.type)
        
        model_inputs = self.converter(atoms)
        model_results = self.nnpot(model_inputs)
        
        return model_results['energy'].cpu().data.numpy()[0]
        

    def predict_energies(self, atoms_list):
        self.nnpot.eval()
        self.nnpot.to(device=self.prediction_device.type)        
        
        model_inputs = self.converter(atoms_list)
        model_results = self.nnpot(model_inputs)

        return model_results['energy'].cpu().data.numpy()
            

    def predict_uncertainty(self, atoms=None, X=None, k=None):
        self.writer('Uncertainty not implemented.')
        return 0.

        
    def predict_local_energy(self, atoms=None, X=None):
        self.writer('Local energy not implemented.')
        return np.zeros((len(atoms), ))


    def predict_forces(self, atoms, return_uncertainty=False, **kwargs):
        self.nnpot.eval()
        self.nnpot.to(device=self.prediction_device.type)
        
        model_inputs = self.converter(atoms)
        model_results = self.nnpot(model_inputs)

        return model_results['forces'].cpu().data.numpy()


    @agox_writer
    def train_model(self, training_data, **kwargs):
        self.writer('Training PaiNN model')
        self.nnpot.train()
        self.nnpot.to(device=self.training_device.type)

        self.ready_state = True
        self.atoms = None

        self.add_data(training_data)

        # Dataloader
        dataset = AtomsDataModule(self.data_path, **self.dataset_settings)
        dataset.prepare_data()
        dataset.setup()

        
        trainer = pl.Trainer(
            accelerator=self.training_device.type,
            # devices=1,
            callbacks=self.callbacks,
            logger=self.logger,
            default_root_dir=str(self.train_path),
            max_epochs=self.trainer_iteration*self.max_epochs_per_iteration,
            max_steps = self.trainer_iteration*self.max_steps_per_iteration,
            enable_progress_bar=False,
            enable_model_summary=False,
            
        )
        ckpt_path = str(self.ckpt_path) if self.trainer_iteration > 1 else None
        trainer.fit(self.task, datamodule=dataset, ckpt_path=ckpt_path)
        
        self.trainer_iteration += 1



            
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
        self.writer(f'length all data: {len(all_data)}')
        
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
        

    
    def load(self, path):
        state_dict = torch.load(path, map_location=self.prediction_device.type).state_dict()
        self.nnpot.load_state_dict(state_dict)
        


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


