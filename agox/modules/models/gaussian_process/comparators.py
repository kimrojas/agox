import numpy as np

from copy import copy

#from dscribe.descriptors import SOAP
#from dscribe.kernels import REMatchKernel


def cosine_dist(f1,f2):
    norm1 = np.linalg.norm(f1)
    norm2 = np.linalg.norm(f2)
    distance = np.sum(np.array(f1)*np.array(f2))/(norm1*norm2)
    
    cos_dist = 0.5*(1-distance)
        
    return cos_dist


def norm2_dist(f1,f2):
    distance = np.linalg.norm(np.array(f1)-np.array(f2))
    
    return distance


def norm1_dist(f1,f2):
    distance = np.sum(abs(np.array(f1)-np.array(f2)))

    return distance


class BagOfBonds(object):

    def __init__(self, n_top=None, pair_cor_cum_diff=0.015,
                 pair_cor_max=0.7, dE=0.25, mic=True, excluded_types=[],
                 overwrite=True, cov_dists=None, normed=True):
        self.pair_cor_cum_diff = pair_cor_cum_diff
        self.pair_cor_max = pair_cor_max
        self.dE = dE
        self.n_top = n_top or 0
        self.mic = mic
        self.excluded_types = excluded_types
        self.overwrite = overwrite
        self.cov_dists = cov_dists
        self.normed = normed

    def get_feature(self,a):
        """ Utility method used to calculate interatomic distances
            returned as a dict sorted after atomic type
            (e.g. (6,6), (6,42), (42,42)). 
        """
        if not self.overwrite:
            if 'features' in a.info:
                return a.info['features']

        #atoms = a[-self.n_top:] This doesnt work, not sure why
        atoms = a

        unique_types = sorted(list(set(atoms.numbers)))
        unique_types = [u for u in unique_types if u not in self.excluded_types]
        pair_cor = {}
        for idx, u1 in enumerate(unique_types):
            i_u1 = [i for i in range(len(atoms)) if atoms[i].number == u1]
            for u2 in unique_types[idx:]:
                i_u2 = [i for i in range(len(atoms)) if atoms[i].number == u2]
                d = []
                if u1 == u2:
                    for i, n1 in enumerate(i_u1):
                        for n2 in i_u2[i+1:]:
                            d.append(float(u1*u2/atoms.get_distance(n1,n2,self.mic)))
                else:
                    for i, n1 in enumerate(i_u1):
                        for n2 in i_u2:
                            d.append(float(u1*u2/atoms.get_distance(n1,n2,self.mic)))

                d.sort()
                if self.cov_dists is not None:
                    d = [np.exp(-di/self.cov_dists[(u1,u2)]) for di in d]
                if len(d) == 0:
                    continue
                pair_cor[(u1,u2)] = d

        a.info['features'] = pair_cor
        return pair_cor

    def get_distance(self, f1, f2, distance='euclidean', normed=False):
        """ Method for calculating the similarity between two objects with features
        f1 and f2, respectively.
        """
        # Calculate similarity.
        d1 = np.hstack([x for (y,x) in sorted(zip(f1.keys(), f1.values()))])
        d2 = np.hstack([x for (y,x) in sorted(zip(f2.keys(), f2.values()))])
        
        df = np.abs(np.array(d1)-np.array(d2))
        
        if distance is 'euclidean':
            cum_diff = np.sqrt(np.sum(df**2))
            return cum_diff
        else:
            cum_diff = np.sum(df)
            if self.normed or normed:
                d_norm = np.sum(np.mean([d1,d2],axis=0))
                return cum_diff / d_norm
            else:
                return cum_diff

    def looks_like(self, a1, a2):
        # Energy criterium
        try:
            dE = abs(a1.get_potential_energy() - a2.get_potential_energy())
            if dE >= self.dE:
                return False
        except:
            pass
            
        # Structure criterium
        f1 = copy(self.get_features(a1))
        f2 = copy(self.get_features(a2))
        if self.cov_dists is not None:
            for key in f1.keys():
                f1[key] = [-np.log(f)*self.cov_dists[key] for f in f1[key]]
                f2[key] = [-np.log(f)*self.cov_dists[key] for f in f2[key]]

        d1 = np.hstack(zip(*sorted(f1.items()))[1])
        d2 = np.hstack(zip(*sorted(f2.items()))[1])
        
        max_d = max(np.abs(np.array(d1)-np.array(d2)))
        s = self.get_similarity(f1,f2,distance='manhattan',normed=True)

#        print s, max_d
        if s > self.pair_cor_cum_diff or max_d > self.pair_cor_max:
            return False
        else:
            return True


# class SOAPclass:

#     def __init__(self,
#                  species,
#                  periodic = True,
#                  rcut = 6.0,
#                  nmax = 8,
#                  lmax = 8,
#                  sigma = 0.3,
#                  eta = 2, # Might need to check this
#                  overwrite = True):
#         self.overwrite = overwrite
#         self.eta = eta
#         self.soap = SOAP(species = species,
#                          nmax = nmax,
#                          lmax = lmax,
#                          rcut = rcut,
#                          sigma = sigma,
#                          average = True)
# #                         average = False)
#         self.kernel = REMatchKernel(alpha = 0.1)

#     def get_features(self,atoms):
#         '''
#         '''
#         if not self.overwrite:
#             if 'features' in atoms.info:
#                 return atoms.info['features']

#         feature = self.soap.create(atoms).astype(np.float64)
#         feature = feature / np.linalg.norm(feature, axis = 1)
#         atoms.info['features'] = feature
#         return feature

#     def get_similarity_kernel(self, f1, f2):
#         """ Method for calculating the similarity between two objects with features
#         f1 and f2, respectively.
#         """
#         local_similiarities = self.kernel.create([f1], [f2]).astype(np.float64)
#         dist = self.kernel.get_global_similarity(local_similiarities)
#         return dist

#     def get_similarity(self, f1, f2):
#         """ Method for calculating the similarity between two objects with features
#         f1 and f2, respectively.
#         """
#         sim = np.dot(f1, f2.T)[0][0]
#         return 1 - sim
