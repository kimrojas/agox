from agox.modules.collectors.collector_std import StandardCollector
from ase.io import write

class TimeDependentCollector(StandardCollector):

    name = 'TimeDependantCollector'

    def num_samples_this_episode(self):
        # self.num_samples must have this form: {0: [], 500: []}
        assert type(self.num_samples) == dict, 'num_samples must be a dict'

        keys = list(self.num_samples.keys())
        keys.sort()
        episode = self.get_episode_counter()

        num_samples = self.num_samples[0] # yes, it must contain 0
        # now step through the sorted list (as long as the episode is past the key) and extract the num_samples
        # the last one extracted will be the most recent num_samples enforced and should apply to this episode
        for k in keys:
            if episode < k:
                break
            num_samples = self.num_samples[k]

        print('NUM_SAMPLES IN EPISODE',episode,':', num_samples)
        return num_samples

