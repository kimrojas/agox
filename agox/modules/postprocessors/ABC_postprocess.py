from abc import ABC, abstractmethod
from agox.modules.helpers.writer import header_footer, Writer
from agox.observer import Observer

import functools

class PostprocessBaseClass(ABC, Observer, Writer):

    def __init__(self, gets={'get_key':'candidates'}, sets={'set_key':'candidates'}, order=3, verbose=True, use_counter=True, prefix=''):
        Observer.__init__(self, gets=gets, sets=sets, order=order)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)

        self.add_observer_method(self.postprocess_candidates, sets=self.sets[0], gets=self.gets[0], order=self.order[0])

    @property
    @abstractmethod
    def name(self):
        return NotImplementedError

    def update(self):
        """
        Used if the postprocessor needs to continously update, e.g. the training of a surrogate potential. 
        """
        pass

    @abstractmethod
    def postprocess(self, candidate):
        """
        Method that actually do the post_processing
        """
        return postprocessed_candidate

    def process_list(self, list_of_candidates):
        """
        This allows all postproccesors to act on a list of candidates serially.
        This function can be overwritten by sub-class to implement parallelism. 
        """
        processed_candidates = []
        for candidate in list_of_candidates:
                processed_candidate = self.postprocess(candidate)
                processed_candidates.append(processed_candidate)
        return processed_candidates

    def assign_from_main(self, main):
        super().assign_from_main(main)
    
    def immunity_decorator(func):
        @functools.wraps(func)
        def wrapper(self, candidate):
            if candidate is None: 
                return None
            if candidate.get_postprocess_immunity():
                return candidate
            else:
                return func(self, candidate)
        return wrapper
    
    def immunity_decorator_list(func):
        @functools.wraps(func)
        def wrapper(self, candidates):
            non_immune_candidates = []
            immune_candidates = []
            for candidate in candidates:
                if not candidate.get_postprocess_immunity():
                    non_immune_candidates.append(candidate)
                else:
                    immune_candidates.append(candidate)

            if len(non_immune_candidates) > 0:
                return func(self, non_immune_candidates) + immune_candidates
            else:
                return immune_candidates
        return wrapper

    def __add__(self, other):
        return SequencePostprocess(processes=[self, other], order=self.order)

    @header_footer
    def postprocess_candidates(self):    
        candidates = self.get_from_cache(self.get_key)
        candidates = self.process_list(candidates)
        
        # Add data in write mode - so overwrites! 
        self.add_to_cache(self.set_key, candidates, mode='w')

    def assign_from_main(self, main):
        super().assign_from_main(main)

class SequencePostprocess(PostprocessBaseClass):

    name = 'PostprocessSequence'

    def __init__(self, processes=[], order=None):
        self.processes = processes
        self.order = order

    def postprocess(self, candidate):
        for process in self.processes:
            candidate = process.postprocess(candidate)

        return candidate

    def process_list(self, list_of_candidates):
        for process in self.processes:
            list_of_candidates = process.process_list(list_of_candidates)
        return list_of_candidates

    def __add__(self, other):
        self.processes.append(other)
        return self
    
    def assign_from_main(self, main):
        for process in self.processes:
            process.assign_from_main(main)

    def attach(self, main):
        for j, process in enumerate(self.processes):
            process.update_order(process.postprocess_candidates, order=self.order[0]+j*0.1)
            process.attach(main)