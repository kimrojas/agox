from abc import ABC, abstractmethod
from agox.observer_handler import Observer

import functools

class PostprocessBaseClass(ABC, Observer):

    def __init__(self, verbose=False, gets={'get_key':'candidates'}, sets={'set_key':'candidates'}, order=3):
        super().__init__(gets=gets, sets=sets, order=order)
        self.verbose = verbose

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
        from agox.modules.postprocessors.postprocess_sequence import PostprocessSequence
        return PostprocessSequence(processes=[self, other])

    def postprocess_candidates(self):    
        candidates = self.get_from_cache(self.get_key)
        candidates = self.process_list(candidates)
        
        # Add data in write mode - so overwrites! 
        self.add_to_cache(self.set_key, candidates, mode='w')

    def attach(self, main, order_offset=0):
        main.attach_observer(self.name+'.postprocess_candidates', self.postprocess_candidates, order=self.order+order_offset)

    def assign_from_main(self, main):
        super().assign_from_main(main)
