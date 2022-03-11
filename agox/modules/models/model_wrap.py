from .model_ABC import ModelBaseClass

def get_wrapped_model_calculator(calc_class):
    """
    Unlike the other models you cannot import the model-object directly, but rather import this function, 
    give it the calculator-class as an argument, which yields an appropriate WrapModel-instantiator. 

    Example: 
    calc = get_wrapped_model_calculator(calc_class)(*calc_args, **calc_kwargs)
    Might be best to only use **calc_kwargs. 
    """

    global WrapModel

    class WrapModel(ModelBaseClass, calc_class):

        name = 'WrappedModel'        

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.set_ready_state(True)

        def train_model(self, training_data, energies):
            """
            The Wrapped Model has no trainable parameters, so there is no training. 
            """
            pass

    return WrapModel
