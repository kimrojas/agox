from agox.observer_handler import Observer

class ObserverWithIO(Observer):

    name = 'ObserverWithIO'

    def __init__(self, gets={'key_name':'candidates'}, sets={'set_name':'seen_candidates'}, order=0):
        super().__init__(gets=gets, sets=sets, order=order)

    def method_that_looks_at_candidates(self):

        candidates = self.main_get_from_cache(self.key_name)

        print(candidates)

        self.main_add_to_cache(self.set_name, candidates, mode='a')


    def attach(self, main):
        main.attach_observer(self.name + 'io_method', self.method_that_looks_at_candidates, order=self.order)