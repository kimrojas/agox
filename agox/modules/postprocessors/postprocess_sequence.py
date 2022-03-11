from agox.modules.postprocessors.postprocess_ABC import PostprocessBaseClass

class PostprocessSequence(PostprocessBaseClass):

    name = 'PostprocessSequence'

    def __init__(self, processes=[]):
        self.processes = processes

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
