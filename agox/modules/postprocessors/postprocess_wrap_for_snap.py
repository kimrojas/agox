from agox.modules.postprocessors.postprocess_ABC import PostprocessBaseClass

class Wrapper(PostprocessBaseClass):

    name = 'WrapperForSnapper'

    def __init__(self, scale):
        self.scale = scale

    @PostprocessBaseClass.immunity_decorator
    def postprocess(self, candidate):
        candidate.positions += self.scale/2
        candidate.wrap()
        candidate.positions -= self.scale/2
        return candidate
        
