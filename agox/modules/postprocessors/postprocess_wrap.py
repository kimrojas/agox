from agox.modules.postprocessors.postprocess_ABC import PostprocessBaseClass

class Wrapper(PostprocessBaseClass):

    name = 'WrapperTheRapper'

    @PostprocessBaseClass.immunity_decorator
    def postprocess(self, candidate):
        candidate.wrap()
        return candidate
        