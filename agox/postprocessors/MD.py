from ase import units
import numpy as np

from ase.constraints import FixAtoms
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation, Stationary
from ase.units import fs, kB

from agox.postprocessors.ABC_postprocess import PostprocessBaseClass



class MDPostprocess(PostprocessBaseClass):

    name = 'MDPostprocessor'

    def __init__(
            self,
            model,
            thermostat=NVTBerendsen,
            start_md=10,
            thermostat_kwargs={'timestep':1.*fs, 'temperature_K':10, 'taut':50*fs},
            prepare_candidate_cls=[MaxwellBoltzmannDistribution, ZeroRotation, Stationary],
            prepare_candidate_kwargs=[{'temperature_K':10}, {}, {}],
            temperature_scheme={10: 20, 30: 50, 5: 20},
            constraints=[],
            fix_template=True,
            verbose=True,
            **kwargs
    ):

        super().__init__(**kwargs)

        self.model = model
        self.start_md = start_md
        self.thermostat = thermostat
        self.thermostat_kwargs = thermostat_kwargs
        self.prepare_candidate_cls = prepare_candidate_cls
        self.prepare_candidate_kwargs = prepare_candidate_kwargs
        self.temperature_scheme = temperature_scheme

        # Constraints:
        self.constraints = constraints
        self.fix_template = fix_template
        self.verbose = verbose
        

    def postprocess(self, candidate):
        candidate.set_calculator(self.model)        
        self.apply_constraints(candidate)

        self.run(candidate)
        
        candidate.add_meta_information('description', self.name)

        self.remove_constraints(candidate)
        return candidate

    def run(self, candidate):
        for cls, kwargs in zip(self.prepare_candidate_cls, self.prepare_candidate_kwargs):
            cls(candidate, **kwargs)
        
        dyn = self.thermostat(candidate, **self.thermostat_kwargs)
        
        if self.verbose:
            dyn.attach(self.write_observer, interval=10, c=candidate)
            
        for temp, steps in self.temperature_scheme.items():
            self.writer(f'MD at {temp}K for {steps} steps.')
            dyn.set_temperature(temp)
            dyn.run(steps)

    def write_observer(self, c):
        self.writer(f'K={c.get_kinetic_energy()}, U={c.get_potential_energy()}')

    def do_check(self, **kwargs):
        if self.get_iteration_counter() > self.start_md and self.model.ready_state:
            return True
        else:
            return False

    def apply_constraints(self, candidate):
        constraints = [] + self.constraints
        if self.fix_template:
            constraints.append(self.get_template_constraint(candidate))

        for constraint in constraints:
            if hasattr(constraint, 'reset'):
                constraint.reset()

        candidate.set_constraint(constraints)

    def remove_constraints(self, candidate):
        candidate.set_constraint([])

    def get_template_constraint(self, candidate):
        return FixAtoms(indices=np.arange(len(candidate.template)))
        
