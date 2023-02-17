import time
import warnings
from scipy.optimize import minimize, OptimizeResult
import numpy as np
from ase.parallel import paropen
from scipy.special import expit
import copy
from scipy.linalg import eigh
from scipy.linalg import solve_triangular, cho_factor, cho_solve


class HyperparameterFitter:       # I introduced fit_prior. everywhere next to fit_weight
    
    @classmethod
    def fit(cls, model, params_to_fit, fit_weight=True, fit_prior=True, pd=None,
            bounds=None, tol=1e-2, txt='mll.txt'):
        
                
        '''
        Fit hyperparameters that are allowed to fit
        based on maximum log likelihood of the data in gp.
        '''
        #gp = model.gp
        
        gp = model

        txt = paropen(txt, 'a')
        txt.write('\n{:s}\n'.format(20 * '-'))
        txt.write('{}\n'.format(time.asctime()))
        txt.write('Number of training points: {}\n'.format(len(gp.X)))
                                                                           ###############   modify pd here.    
        arguments = (model, params_to_fit, fit_weight, fit_prior, pd, txt)

        params = []

        # In practice, we optimize the logarithms of the parameters:
        for string in params_to_fit:
            params.append(np.log10(gp.hp[string]))

        t0 = time.time()

        result = minimize(cls.neg_log_likelihood,
                          params,
                          args=arguments,
                          method='Nelder-Mead',
                          options={'fatol': tol})
        


        txt.write("Time spent minimizing neg log likelihood: "
                  "{:.02f} sec\n".format(time.time() - t0))

        converged = result.success

        # collect results:
        optimalparams = {}
        powered_results = np.power(10, result.x)
        
        
        
        for p, pstring in zip(powered_results, params_to_fit):
            optimalparams[pstring] = p

        gp.set_hyperparams(optimalparams)
        gp.train(gp.X, gp.Y)

        txt.write('{} success: {}\n'.format(str(gp.hp), converged))
        txt.close()

        return model



    @staticmethod
    def logP(gp):
        y = gp.Y.flatten()
        print('LOOOK AT MEEEEE')
        print(y)
        
        
        logP = (- 0.5 * np.dot(y - gp.prior_array, gp.model_vector)
                - np.sum(np.log(np.diag(gp.L)))
                - len(y) / 2 * np.log(2 * np.pi))
        
        return logP


    @staticmethod
    def neg_log_likelihood(params, *args, fit_weight=True, fit_prior=True):

        model, params_to_fit, fit_weight, fit_prior, prior_distr, txt = args
        #gp = model.gp
        gp = model


        params_here = np.power(10, params)


        txt1 = ""
        paramdict = {}
        for p, pstring in zip(params_here, params_to_fit):
            paramdict[pstring] = p
            txt1 += "{:18.06f}".format(p)

        gp.set_hyperparams(paramdict)
        gp.train(gp.X, gp.Y)
        
        
        if fit_prior:                                       
            PriorFitter.fit(model)
            
            
        if fit_weight:
            GPPrefactorFitter.fit(model)


        # Compute log likelihood
        #logP = gp.logP()
        logP = HyperparameterFitter.logP(gp)
        
        # Prior distribution:
        if prior_distr is not None:
            logP += prior_distr.get(gp.hp['scale'])


        # Don't let ratio fall too small, resulting in numerical
        # difficulties:
        if 'ratio' in params_to_fit:
            ratio = params_here[params_to_fit.index('ratio')]
            if ratio < 1e-6:
                logP -= (1e-6 - ratio) * 1e6

        txt.write('Parameters: {:s}       -logP: {:12.02f}\n'
                  .format(txt1, -logP))
        txt.flush()
              
        return -logP



class PriorFitter:
    
    @staticmethod
    def fit(model):
        gp=model#.gp
            
        gp.prior.update(gp.X, gp.Y, gp.L)
        gp.train(gp.X, gp.Y)
        #########
        # CL. Added by me. prior_array and model_vector was not re-calculated after prior update
        #prior_array = gp.calculate_prior_array(gp.X)
        #model_vector = gp.Y.flatten() - prior_array
        #gp.model_vector=cho_solve((gp.L, gp.lower), model_vector, overwrite_b=True, check_finite=True)
        return model



class GPPrefactorFitter:

    @staticmethod
    def fit(model):
                
        gp = model#.gp

        oldvalue = gp.hp['weight']
        
        y = np.array(gp.Y).flatten()
              
        factor = np.sqrt(np.dot(y - gp.prior_array, gp.model_vector) / len(y))
        
        newvalue = factor * oldvalue

        gp.set_hyperparams({'weight': newvalue})
        gp.set_hyperparams({'noise': gp.hp['ratio']*newvalue})

        # Rescale accordingly ("re-train"):
        gp.model_vector /= factor**2            # Andreas har ikke dette, da han tager prefaktoren udenfor K0
        gp.L *= factor  # lower triangle of Cholesky factor matrix
        gp.K *= factor**2
        
        
        
        return model



class InfPrior():
    
    def __init__(self, forbidden_value, lower):
        self.name='InfPrior'
        self.forbidden_value=forbidden_value
        self.lower=lower   
        
    def get(self,x):
        if self.lower and x<self.forbidden_value: 
            value=-np.inf
        elif (not self.lower) and x>self.forbidden_value: 
            value=-np.inf
        else: 
            value=0
        return value
    


class PriorDistributionSigmoid():

    def __init__(self, s_pen=100, s_free=100, s=0.99, factor=1):
        ''' Sigmoid Prior "distribution" for
        hyperparameter 'scale'. Parameters correspond to
        parameters of scipy.special.expit. '''
        
        self.name='Sigmoid'
        self.s=s
        self.factor=factor
        self.llmax=1
        
        self.get_sig_params(s_pen,s_free)
        
    def get(self, x):
        
        y = self.rate*(x - self.loc)

        # Filter out warnings regarding
        # close-to-zero values in log function
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            value = self.factor*np.log(expit(y))

        return value
    
    def update_factor(self, ll):
        #g*f/F=1 --> g=F/f        
        y=-self.rate*self.loc
        f=np.log(expit(y))
        self.factor=ll/f
    
    
    def update(self, fingerprint_distances):
   
        s_pen=min(fingerprint_distances)
        s_free=np.mean(fingerprint_distances)
        
        self.get_sig_params(s_pen, s_free)
                
    #@staticmethod
    def get_sig_params(self, s_pen, s_free):
        width=(s_free-s_pen)/2
        self.loc=s_pen+width
        self.rate=(-1/width)*np.log((1-self.s)/self.s)
        
        
class PriorDistributionDoubleSigmoid():
    
    def __init__(self, s1_pen=100, s1_free=100, s2_pen=100, s2_free=100, s1_factor=1, s2_factor=1, s=0.99):
        
        self.name='DoubleSigmoid'
        
        self.s=s
        
        self.s1_factor=s1_factor
        
        self.s2_factor=s2_factor
        
        self.sig1=PriorDistributionSigmoid(s1_pen, s1_free)
        
        self.sig2=PriorDistributionSigmoid(s2_pen, s2_free)


    def get(self, x):
        
        val1=self.s1_factor*self.sig1.get(x)
        
        val2=self.s2_factor*self.sig2.get(x)
        
        value=val1*val2
        
        return value
    
    def update_factors(self,ll):
        
        self.sig1.update_factor(ll)
        self.sig2.update_factor(ll)
    
        
    def update(self, fingerprint_distances):
 
        s_free=np.mean(fingerprint_distances)
        
        s1_pen=min(fingerprint_distances)
         
        s2_pen=max(fingerprint_distances)
        
        self.sig1.get_sig_params(s1_pen, s_free)
        self.sig2.get_sig_params(s2_pen, s_free)
    
    

class PriorDistributionLogNormal():

    def __init__(self, loc=6, width=1, log=True, update_width=True):
        ''' LogNormal "distribution" for
        hyperparameter 'scale'. Parameters correspond
        to mean (location of peak) and standard deviation
        of the Log_Nmormal in logaritmic space. 
        in non-logaritmic space the distribution mode is located at exp(loc-width**2)
        the prior is set up such that the mode is always located at the mean value
        of feature space distances in non-logaritmic space. 
        '''
        self.name='LogNormal'
        self.loc=loc
        self.width=width
        self.log=log
        self.update_width=update_width
        
    def get(self,x):
        
        log_fp_dist_std=self.width
        
        mode=self.loc

        log_fp_dist_mean=mode+log_fp_dist_std**2
        
        A=-0.5*( (np.log(x) - (log_fp_dist_mean-log_fp_dist_std**2) ) /log_fp_dist_std)**2
        
        B=- log_fp_dist_mean + 0.5*log_fp_dist_std**2 - np.log(log_fp_dist_std*np.sqrt( 2*np.pi ))
        
        log_scale_prior= A+B
        
        return log_scale_prior
    
    def update(self, fingerprint_distances):   
        #toppoint=0.5*(min(fingerprint_distances)+max(fingerprint_distances))
        toppoint=np.mean(fingerprint_distances)
        maxpoint=max(fingerprint_distances)
        self.loc=np.log(toppoint)        
        
        if self.update_width:
            if self.log:
                self.width=0.5*(np.log(maxpoint)-np.log(toppoint))        
            else:
                self.width=np.log(0.5*(maxpoint-toppoint))