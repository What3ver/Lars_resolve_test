"""
resolve.py
Written by Henrik Junklewitz

RESOLVE is...

Copyright 2014 Henrik Junklewitz

RESOLVE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RESOLVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with RESOLVE. If not, see <http://www.gnu.org/licenses/>.

"""

from __future__ import division
from time import time
from subprocess import call

#import necessary modules
import pylab as pl
import numpy as np
from nifty import *
import Messenger as M
import general_response as r
from general_IO import read_data_from_ms
from nifty import nifty_tools as nt
from casa import ms as mst
from casa import image as ia 
import casa

#a few global constants
q = 1e-15
C = 299792458
asec2rad = 4.84813681e-6



def resolve(ms, imsize, cellsize, algorithm = 'ln-map', init_type_s = 'dirty',\
    use_init_s = False, init_type_p = 'k-2_mon', lastit = None, freq = [0,0] ,\
    pbeam = None, uncertainty = False, noise_est = None, map_algo = 'sd', \
    pspec_algo = 'cg', barea = 1, map_conv = 1e-1, pspec_conv = 1e-1, \
    save = None, callback = 3, plot = False, simulating = False, **kwargs):

    """
        A RESOLVE-reconstruction.
    
        Args:
            ms: Measurement set that holds the data to be imaged.
            imsize: Size of the image.
            cellsize: Size of one grid cell.
            algorithm: Which algorithm to be used. 
                1) 'ln-map': standard-resolve 
                2) 'wf': simple Wiener Filter 
                3) 'Gibbs': Gibbs-Energy minimizer
                4) 'samp': Sampling
            init_type_s: What input should be used to fix the monopole of the map
                by effectively estimating rho_0.
                1) 'dirty'
                2) 'user-defined'
            use_init_s: Whether to use the init_type_s as a starting guess \
                (default is False and uses a constant map close to zero).
            init_type_p: Starting guess for the power spectrum.
                1) 'k^2': Simple k^2 power spectrum.
                2) 'k^2_mon': Simple k^2 power spectrum with fixed monopole.
                3) 'zero': Zero power spectrum.
            lastit: Integer n or None. Whether to start with iteration n.
            freq: Whether to perform single band or wide band RESOLVE.
                1) [spw,cha]: single band
                2) 'wideband'
            pbeam: user-povided primary beam pattern.
            uncertainty: Whether to attempt calculating an uncertainty map \
                (EXPENSIVE!).
            noise_est: Whether to take the measure noise variances or make an \
                estimate for them.
                1) 'simple': simply try to estimate the noise variance using \
                    the rms in the visibilties.
                2) 'ecf': try a poor-man's ECF. WARNING: Changes the whole \
                    algorithm and makes RESOLVE considerably slower.
            map_algo: Which optimization algorithm to use for the signal estimate.\
                1) 'sd': steepest descent, recommended for robustness.
                2) 'lbfgs'
            pspec_algo: Which optimization algorithm to use for the pspec estimate.\
                1) 'cg': conjugate gradient, recommended for speed.
                2) 'sd': steepest descent, only for robustness.
            barea: if given a number, all units will be converted into
                units of Jy/beam for a choosen beam. Otherwise the units
                of images are always in Jy/px.
            map_conv: stop criterium for RESOLVE in map reconstruction.
            pspec_conv: stop criterium for RESOLVE in pspec reconstruction.
            save: If not None, save all iterations to disk using the given
                base name.
            callback: If given integer n, save every nth sub-iteration of \
                intermediate optimization routines.
            plot: Interactively plot diagnostical plots. Only advised for testing.
            simulate: Whether to simulate a signal or not. 
        
        kwargs:
            Set numerical or simulational parameters. All are set to tested and \
            robust default values. For details see internal functions numparameters
            and simparameters.
            
        Returns:
            m: The MAP solution
            p: The power spectrum.
            u: Signal uncertainty (only if wanted).
    """


    # Turn off nifty warnings to avoid "infinite value" warnings
    about.warnings.off()
    
    # Define parameter class

    
    params = parameters(ms, imsize, cellsize, algorithm, init_type_s, \
                        use_init_s, init_type_p, lastit, freq, pbeam, \
                        uncertainty, noise_est, map_algo, pspec_algo, \
                        barea, map_conv, pspec_conv, save, callback, plot)
                        
    numparams = numparameters(kwargs)
    
    if simulating:
        
        simparams = simparameters(params, kwargs)
                                                    
    # Prepare a number of diagnostics if requested
    if params.plot:
        pl.ion()
    else:
        pl.ioff()
    if params.save:
        if not os.path.exists('general_output'):
            os.makedirs('general_output')
        if not os.path.exists('last_iterations'):
            os.makedirs('last_iterations')
        if not os.path.exists('m_reconstructions'):
            os.makedirs('m_reconstructions')
        if not os.path.exists('p_reconstructions'):
            os.makedirs('p_reconstructions')
        if not os.path.exists('D_reconstructions'):
            os.makedirs('D_reconstructions')
        logfile = 'general_output/' + params.save + '.log'
    else:
        logfile = None
    logger = M.Messenger(verbosity=2, add_timestamp=False, logfile=logfile)

    if params.save:
        params.printparameters('general_output')
        numparams.printparameters('general_output')
        if simulating:
            simparams.printparameters('general_output')

    logger.header1('Starting Bayesian total intensity reconstruction.')

    # data setup
    if simulating:
        d, N, R, di, d_space, s_space = simulate(params, simparams, logger)
    else:
        d, N, R, di, d_space, s_space = datasetup(params, logger)
    
    # Begin: Starting guesses for m *******************************************

    # estimate rho0, the constant part of the lognormal field rho = rho0 exp(s)
    # effectively sets a different prior mean so that the field doesn't go to 
    # zero in first iterations     
    if init_type_s == 'dirty':
        mtemp = field(s_space, target=s_space.get_codomain(), val=di)

    else:
        # Read-in userimage and convert to Jy/px
        mtemp = field(s_space, target=s_space.get_codomain(), \
                      val=np.load('userimage.npy')/params.barea)

    
    rho0 = np.mean(mtemp.val[np.where(mtemp.val>=np.max(mtemp.val) / 4)])
    logger.message('rho0: ' + str(rho0))
    if rho0 < 0:
        logger.warn('Monopole level of starting guess negative. Probably due \
            to too many imaging artifcts in userimage')
        
    # Starting guess for m, either constant close to zero, or lastit from
    # a file with save-basis-string 'save', or directly from the user
    if lastit == None:
        m_I = field(s_space, val = numparams.m_start)
    elif lastit != None:
        logger.message('using last m-iteration from previous run.')
        m_I = field(s_space, val = np.load(params.save + str(lastit) + \
            "_m.npy"))
    elif use_init_s:
        m_I = field(s_space, val = np.log(np.abs(mtemp)))

    # Begin: Starting guesses for pspec ***************************************

    # Basic k-space
    k_space = R.domain.get_codomain()
        
    #Adapts the k-space properties if binning is activated.
    if not numparams.bin is None:
        k_space.set_power_indices(log=True, nbins=numparams.bin)
    
    # k-space prperties    
    kindex,rho_k,pindex,pundex = k_space.get_power_indices()

    # Simple k^2 power spectrum with p0 from numpars and a fixed monopole from
    # the m starting guess
    if params.init_type_p == 'k^2_mon':
        pspec = np.array((1+kindex)**-2 * numparams.p0)
        pspec_mtemp = mtemp.power(pindex=pindex, kindex=kindex, rho=rho_k)
        #see notes, use average power in dirty map to constrain monopole
        pspec[0] = (np.prod(k_space.vol)**(-2) * np.log(\
            np.sqrt(pspec_mtemp[0]) *  np.prod(k_space.vol))**2) / 2.
    # zero power spectrum guess 
    elif params.init_type_p == 'zero':
        pspec = 0
    # power spectrum from last iteration 
    elif lastit != None:
        logger.message('using last p-iteration from previous run.')
        pspec = np.load(params.save + str(lastit) + "_p.npy")
    # default simple k^2 spectrum with free monopole    
    else:
        pspec = np.array((1+kindex)**-2 * numparams.p0) 
 
    # diagnostic plot of m starting guess
    if params.save:
        save_results(np.exp(m_I.val),"exp(Starting guess)",\
            "m_reconstructions/" + params.save + "_expm0", rho0 = rho0)

            
    # Begin: Start Filter *****************************************************

    if params.algorithm == 'ln-map':
        
        logger.header1('Starting RESOLVE.')
        
        if params.freq is not 'wideband':
            #single band I-Filter
            t1 = time()
            m, p = mapfilter_I(d, m_I, pspec, N, R, logger, rho0, k_space, \
                params, numparams)
            t2 = time()
            
        else:
            logger.failure('Wideband Resolve not yet implemented')
            
    elif params.algorithm == 'wf':
        
        logger.failure('Wiener Filter not yet implemented')
        
    elif params.algorithm =='gibbsenergy':
        
        logger.failure('Gibbs energy filter not yet implemented')
        
    elif params.algorithm == 'sampling':
        
        logger.failure('Sampling algorithm not yet implemented')
        
        

    logger.success("Completed algorithm.")
    logger.message("Time to complete: " + str((t2 - t1) / 3600.) + ' hours.')

    # Begin: Some plotting stuff **********************************************

    if params.save:
        
        save_results(np.exp(m.val),"exp(Solution)",\
            "m_reconstructions/" + params.save + "_expmfinal", rho0 = rho0)
            
        save_results(kindex,"final power spectrum",\
            "p_reconstructions/" + params.save + "_powfinal", value2 = pspec)
        

    # ^^^ End: Some plotting stuff *****************************************^^^

    return m, p

#------------------------------------------------------------------------------


def datasetup(params, logger):
    """
        
    Data setup, feasible for the nifty framework.
        
    """
    
    logger.header2("Running data setup.")

    if params.noise_est == 'simple':
        vis, sigma, u, v, freqs = read_data_from_ms(params.ms, \
            noise_est = True)
    else:
        vis, sigma, u, v, freqs = read_data_from_ms(params.ms)
        
    # wideband imaging
    if params.freq == 'wideband':
        logger.message('Wideband imaging not yet implemented')
    # single-band imaging
    else:
        nspw,chan = params.freq[0], params.freq[1]
        vis = vis[nspw][chan]
        sigma = sigma[nspw][chan]
        u = u[nspw][chan]
        v = v[nspw][chan]
        
    # noise estimation via ecf   
    if params.noise_est == 'ecf':
        logger.message('ECF noise estimate not yet implemented')
    
    variance = sigma**2
    variance[variance<1e-10] = np.mean(variance[variance>1e-10])


    # basic diagnostics
    if params.save:
        # maximum k-mode and resolution of data
        uvrange = np.array([np.sqrt(u[i]**2 + v[i]**2) for i in range(len(u))])
        dx_real_rad = (np.max(uvrange))**-1
        logger.message('resolution\n' + 'rad ' + str(dx_real_rad) + '\n' + \
            'asec ' + str(dx_real_rad/asec2rad))

        save_results(u,'UV', "general_output/" + params.save + "_uvcov", \
            plotpar='o', value2 = v)


    # definition of wideband data operators
    if params.freq == 'wideband':
        logger.message('Wideband imaging not yet implemented')
        
    # definition of single band data operators
    else:
        d_space = point_space(len(u), datatype = np.complex128)
        s_space = rg_space(params.imsize, naxes=2, dist = params.cellsize, zerocenter=True)
        
        # primary beam function
        if not params.pbeam is None:
            logger.message('Calculating VLA beam from Perley fit')
            A = make_primbeam_aips(s_space.dim(split = True)[0], \
                s_space.dim(split = True)[1] , s_space.dist()[0], s_space.dist()[1], \
                1.369)
        else:
            A = 1.
        # response operator
        R = r.response(s_space, sym=False, imp=True, target=d_space, \
                       para=[u,v,A,False])
    
        d = field(d_space, val=vis)

        N = N_operator(domain=d_space,imp=True,para=[variance])
        
        # dirty image from CASA for comparison
        di = make_dirtyimage(params, logger)

        # more diagnostics if requested
        if params.save:
            # plot the dirty beam
            uvcov = field(d_space,val=np.ones(len(u), dtype = np.complex128))
            db = R.adjoint_times(uvcov) * s_space.vol[0] * s_space.vol[1]       
            save_results(db,"dirty beam","general_output/" + params.save + "_db")
            
            # plot the dirty image
            save_results(di,"dirty image","general_output/" + params.save + "_di")

        return  d, N, R, di, d_space, s_space
    


def mapfilter_I(d, m, pspec, N, R, logger, rho0, k_space, params, numparams):
    """
    """

    logger.header1("Begin total intensity filtering")  
      
    s_space = m.domain
    kindex = k_space.get_power_indices()[0]
    
         
    # Sets the alpha prior parameter for all modes
    if not numparams.alpha_prior is None:
        alpha = numparams.alpha_prior
    else:
        alpha = np.ones(np.shape(kindex))
        
    # Defines important operators
    S = power_operator(k_space, spec=pspec, bare=True)
    M = M_operator(domain=s_space, sym=True, imp=True, para=[N, R])
    j = R.adjoint_times(N.inverse_times(d))
    D = D_operator(domain=s_space, sym=True, imp=True, para=[S, M, m, j, \
        numparams.M0_start, rho0, params, numparams])

    
    
    # diagnostic plots
    if params.save:       
        save_results(j,"general_output/" + params.save + 'j','_j', rho0 = rho0)

    # iteration parameters
    convergence = 0
    git = 1
    plist = [pspec]
    mlist = [m]
    

    while git <= numparams.global_iter:
        """
        Global filter loop.
        """
        logger.header2("Starting global iteration #" + str(git) + "\n")

        # update power operator after each iteration step
        S.set_power(newspec=pspec,bare=True)
          
        # uodate operator and optimizer arguments  
        args = (j, S, M, rho0)
        D.para = [S, M, m, j, numparams.M0_start, rho0, params, numparams]

        #run nifty minimizer steepest descent class
        logger.header2("Computing the MAP estimate.\n")

        mold = m
        
        #vorlaeufige Werte fuer u B und NU (fuer die Punktquellen), allgeimeiner und frueher einfuegen 
        u = field(s_space) 
        B = 1.5
        NU = 1.0

        if params.map_algo == 'sd':
            en = energy(args)
            minimize = nt.steepest_descent(en.egg,spam=callbackfunc,note=True)
            m = minimize(x0=m, alpha=numparams.map_alpha, \
                tol=numparams.map_tol, clevel=numparams.map_clevel, \
                limii=numparams.map_iter)[0]

        elif params.map_algo == 'up': #use pointsource NEW up definieren
            args = (j, S, M, rho0, B, NU,m,u)
            en = energy_up(args) 
            print(en.egg_u(u)[0])
            minimize = nt.steepest_descent(en.egg_u,spam=callbackfunc_u,note=True)
            u = minimize(x0=u, alpha=numparams.map_alpha, \
                tol=numparams.map_tol, clevel=numparams.map_clevel, \
                limii=numparams.map_iter)[0]
            en.ueff = u
            minimize = nt.steepest_descent(en.egg_s,spam=callbackfunc_m,note=True)
            m = minimize(x0=m, alpha=numparams.map_alpha, \
                tol=numparams.map_tol, clevel=numparams.map_clevel, \
                limii=numparams.map_iter)[0]
            en.seff = m

        elif params.map_algo == 'lbfgs':
            logger.failure('lbfgs algorithm not yet implemented.')
            
        if params.save:
            if params.map_algo == 'sd':
                save_results(exp(m.val), "map, iter #" + str(git), \
                    "m_reconstructions/" + params.save + "_expm" + str(git), \
                    rho0 = rho0)
            elif params.map_algo == 'up':
                save_results(exp(m.val+u.val), "map, iter #" + str(git), \
                    "mu_reconstructions/" + params.save + "_expmu" + str(git), \
                    rho0 = rho0)        
                save_results(exp(m.val), "map, iter #" + str(git), \
                    "m_reconstructions/" + params.save + "_expm" + str(git), \
                    rho0 = rho0)
                save_results(exp(u.val), "map, iter #" + str(git), \
                    "u_reconstructions/" + params.save + "_expu" + str(git), \
                    rho0 = rho0)

        logger.header2("Computing the power spectrum.\n")

        #extra loop to take care of possible nans in PS calculation
        psloop = True
        M0 = numparams.M0_start
        while psloop:
            
            D.para = [S, M, m, j, M0, rho0, params, numparams]
            
            Sk = projection_operator(domain=k_space)
            #bare=True?
            logger.message('Calculating Dhat.')
            D_hathat = D.hathat(domain=s_space.get_codomain(),\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
            logger.message('Success.')

            pspec = infer_power(m,domain=Sk.domain,Sk=Sk,D=D_hathat,\
                q=1E-42,alpha=alpha,perception=(1,0),smoothness=True,var=\
                numparams.smoothing, bare=True)

            if np.any(pspec == False):
                print 'D not positive definite, try increasing eta.'
                if M0 == 0:
                    M0 += 0.1
                M0 *= 1e6
                D.para = [S, M, m, j, M0, rho0, params, numparams]
            else:
                psloop = False
            
        logger.message("    Current M0:  " + str(D.para[4])+ '\n.')


        mlist.append(m)
        plist.append(pspec)

        
        if params.save:
            save_results(kindex,"ps, iter #" + str(git), \
                "p_reconstructions/" + params.save + "_p" + str(git), \
                value2=pspec)
            
            # powevol plot needs to be done in place
            pl.figure()
            for i in range(len(plist)):
                pl.loglog(kindex, plist[i], label="iter" + str(i))
            pl.title("Global iteration pspec progress")
            pl.legend()
            pl.savefig("p_reconstructions/" + params.save + "_powevol.png")
            pl.close()
        
        # convergence test in map reconstruction
        if np.max(np.abs(m - mold)) < params.map_conv:
            logger.message('Image converged.')
            convergence += 1
        
        # convergence test in power spectrum reconstruction 
        if np.max(np.abs(np.log(pspec)/np.log(S.get_power()))) < np.log(1e-1):
            logger.message('Power spectrum converged.')
            convergence += 1
        
        #global convergence test
        if convergence >= 4:
            logger.message('Global convergence achieved at iteration' + \
                str(git) + '.')
            if params.uncertainty:
                logger.message('Calculating uncertainty map as requested.')
                D_hat = D.hat(domain=s_space,\
                ncpu=numparams.ncpu,nrun=numparams.nrun)
                save_results(D_hat.val,"relative uncertainty", \
                "D_reconstructions/" + params.save + "_D")
                
            return m, pspec

        git += 1
    
    if params.uncertainty:
        logger.message('Calculating uncertainty map as requested.')
        D_hat = D.hat(domain=s_space,\
        ncpu=numparams.ncpu,nrun=numparams.nrun)
        save_results(D_hat.val,"relative uncertainty", \
        "D_reconstructions/" + params.save + "_D")

    return m, pspec


    
#------------------------------------------------------------------------------


class N_operator(operator):
    """
    Wrapper around a standard Noise operator. Handles radio astronomical flags.
    """
    
    def _multiply(self, x):
        
        sigma = self.para[0]
        
        mask = sigma>0
        sigma[sigma==0] = 1.

        Ntemp = diagonal_operator(domain=self.domain, diag = sigma**2)

        return mask * Ntemp(x)

    def _inverse_multiply(self, x):
        
        sigma = self.para[0]
        
        mask = sigma>0
        sigma[sigma==0] = 1.

        Ntemp = diagonal_operator(domain=self.domain, diag = sigma**2)

        return mask * Ntemp.inverse_times(x)
        
        
class M_operator(operator):
    """
    """

    def _multiply(self, x):
        """
        """
        N = self.para[0]
        R = self.para[1]

        return R.adjoint_times(N.inverse_times(R.times(x)))


class D_operator(operator):
    """
    """

    def _inverse_multiply(self, x):
        """
        """

        S = self.para[0]
        M = self.para[1]
        m = self.para[2]
        j = self.para[3]
        M0 = self.para[4]
        A = self.para[5]

        nondiagpart = M_part_operator(M.domain, imp=True, para=[M, m, A])

        diagpartval = (-1. * j * A * exp(m) + A * exp(m) * M(A * exp(m))).hat()
        
        diag = diagonal_operator(domain = S.domain, diag = 1. * M0)

        part1 = S.inverse_times(x)
        part2 = diagpartval(x)
        part3 = nondiagpart(x)
        part4 = diag(x)

        return part1 + part2 + part3 + part4 

    _matvec = (lambda self, x: self.inverse_times(x).val.flatten())
    

    def _multiply(self, x):
        """
        the operator is defined by its inverse, so multiplication has to be
        done by inverting the inverse numerically using the conjugate gradient
        method from scipy
        """
        convergence = 0
        numparams = self.para[7]
        params = self.para[6]

        if params.pspec_algo == 'cg':
            x_,convergence = nt.conjugate_gradient(self._matvec, x, \
                note=True)(tol=numparams.pspec_tol,clevel=numparams.pspec_clevel,\
                limii=numparams.pspec_iter)
                
        elif params.pspec_algo == 'sd':
            x_,convergence = nt.steepest_descent(self._matvec, x, \
                note=True)(tol=numparams.pspec_tol,clevel=numparams.pspec_clevel,\
                limii=numparams.pspec_iter)
                    
        return x_
        


class M_part_operator(operator):
    """
    """

    def _multiply(self, x):
        """
        """
        M = self.para[0]
        m = self.para[1]
        A = self.para[2]

        return 1. * A * exp(m) * M(A * exp(m) * x)



class energy(object):
    
    def __init__(self, args):
        self.j = args[0]
        self.S = args[1]
        self.M = args[2]
        self.A = args[3]
        
    def H(self,x):
        """
        """
        part1 = x.dot(self.S.inverse_times(x.weight(power = 0)))  / 2
        part2 = self.j.dot(self.A * exp(x))
        part3 = self.A * exp(x).dot(self.M(self.A * exp(x))) / 2
        
        
        return part1 - part2 + part3
    
    def gradH(self, x):
        """
        """
#        print 'x', x.domain
#        print 'j', self.j.domain
#        print 'M', self.M.domain
#        print 'S', self.S.domain.get_codomain()
#        print 'codo', self.S.domain.get_codomain()
#        
#        x = field(self.S.domain.get_codomain(), val=x)
    
        temp1 = self.S.inverse_times(x)
        #temp1 = temp1.weight(power=2)
        temp = -self.j * self.A * exp(x) + self.A* exp(x) * \
            self.M(self.A * exp(x)) + temp1
    
        return temp
    
    def egg(self, x):
        
        E = self.H(x)
        g = self.gradH(x)
        
        return E,g

class energy_up(object):
    
    def __init__(self, args):
        self.j = args[0]
        self.S = args[1]
        self.M = args[2]
        self.A = args[3]
        self.B = args[4]
        self.NU = args[5]
        self.seff = args[6]
        self.ueff = args[7]
        self.b = self.B-1

    def H(self,x,u):
        """
        """
        I = exp(x)+exp(u)
        part1 = x.dot(self.S.inverse_times(x.weight(power = 0)))  / 2
        part2 = self.j.dot(self.A * (exp(x)+exp(u)))
        part3 = self.A * I.dot(self.M(self.A * I)) / 2
        part4 = -u.dot(self.b)-(exp(-u)).dot(self.NU) 

        return part1 - part2 + part3 - part4
    
    def gradH_s(self, x,u):
        """
        """
        
        I = exp(x)+exp(u)
        temp1 = self.S.inverse_times(x)
        temp = -self.j * self.A * exp(x) + self.A* exp(x) * \
            self.M(self.A * I) + temp1
    
        return temp

    def gradH_u(self, x,u):
        """
        """
        I = exp(x)+exp(u)
        temp1 = self.b - self.NU * exp(-u)
        temp = -self.j * self.A * exp(u) + self.A* exp(u) * \
            self.M(self.A * I) + temp1
    
        return temp
    
    def egg_s(self, x):
        
        E = self.H(x,self.ueff)
        gs = self.gradH_s(x,self.ueff)
        return E,gs

    def egg_u(self, u):
        
        E = self.H(self.seff,u)
        gu = self.gradH_u(self.seff,u)        
        return E,gu

class parameters(object):

    def __init__(self,ms, imsize, cellsize, algorithm, init_type_s, \
                 use_init_s, init_type_p, lastit, freq, pbeam, \
                 uncertainty, noise_est, map_algo, pspec_algo, barea,\
                 map_conv, pspec_conv, save, callback, plot):
        
        self.para = []

        self.ms = ms
        self.para.append(['ms',str(self.ms)]) 
        self.imsize = imsize
        self.para.append(['imsize',str(self.imsize)]) 
        self.cellsize = cellsize
        self.para.append(['cellsize',str(self.cellsize)]) 
        self.algorithm = algorithm
        self.para.append(['algorithm',str(self.algorithm)]) 
        self.init_type_s = init_type_s 
        self.para.append(['init_type_s',str(self.init_type_s)]) 
        self.use_init_s = use_init_s
        self.para.append(['use_init_s',str(self.use_init_s)]) 
        self.init_type_p = init_type_p
        self.para.append(['init_type_p',str(self.init_type_p)]) 
        self.lastit = lastit
        self.para.append(['lastit',str(self.lastit)]) 
        self.freq = freq
        self.para.append(['freq',str(self.freq)]) 
        self.pbeam = pbeam
        self.para.append(['pbeam',str(self.pbeam)]) 
        self.uncertainty = uncertainty
        self.para.append(['uncertainty',str(self.uncertainty)]) 
        self.noise_est = noise_est
        self.para.append(['noise_est',str(self.noise_est)]) 
        self.map_algo = map_algo
        self.para.append(['map_algo',str(self.map_algo)]) 
        self.pspec_algo = pspec_algo
        self.para.append(['pspec_algo',str(self.pspec_algo)]) 
        self.barea = barea
        self.para.append(['barea',str(self.barea)]) 
        self.save = save
        self.para.append(['save',str(self.save)]) 
        self.map_conv = map_conv
        self.para.append(['map_conv',str(self.map_conv)]) 
        self.pspec_conv = pspec_conv
        self.para.append(['pspec_conv',str(self.pspec_conv)]) 
        self.callback = callback
        self.para.append(['callback',str(self.callback)]) 
        self.plot = plot
        self.para.append(['plot',str(self.plot)]) 
        
        # a few global parameters needed for callbackfunc
        global gcallback
        global gsave
        gcallback = callback
        gsave = save

    def printparameters(self,place):
     
        f = open(place+'/resolve_parameters.txt','w')
        for name in self.para:
            f.write('{0:15}: {1:15}'.format(name[0],name[1]) + '\n')

class numparameters(object):
    
    def __init__(self, kwargs):
    
        self.para = []
         
        if 'm_start' in kwargs:
            self.m_start = kwargs['m_start']
            self.para.append(['m_start',str(self.m_start)]) 
        else: 
            self.m_start = 0.1
            
        if 'global_iter' in kwargs:
            self.global_iter = kwargs['global_iter']
            self.para.append(['global_iter',str(self.global_iter)]) 
        else: 
            self.global_iter = 50
            
        if 'M0_start' in kwargs:
            self.M0_start = kwargs['M0_start']
            self.para.append(['M0_start',str(self.M0_start)]) 
        else: 
            self.M0_start = 0
            
        if 'bins' in kwargs:
           self.bin = kwargs['bins']
           self.para.append(['bin',str(self.bin)]) 
        else:
           self.bin = 70 
           
        if 'p0' in kwargs:
           self.p0 = kwargs['p0']
           self.para.append(['p0',str(self.p0)]) 
        else:
           self.p0 = 1.   
           
        if 'alpha_prior' in kwargs:
           self.alpha_prior = kwargs['alpha_prior']
           self.para.append(['alpha_prior',str(self.alpha_prior)]) 
        else:
           self.alpha_prior = None 
           
        if 'smoothing' in kwargs:
           self.smoothing = kwargs['smoothing']
           self.para.append(['smoothing',str(self.smoothing)]) 
        else:
           self.smoothing = 10.
               
        if 'pspec_tol' in kwargs:
           self.pspec_tol = kwargs['pspec_tol']
           self.para.append(['pspec_tol',str(self.pspec_tol)]) 
        else:
           self.pspec_tol = 1e-3   
           
        if 'pspec_clevel' in kwargs:
           self.pspec_clevel = kwargs['pspec_clevel']
           self.para.append(['pspec_clevel',str(self.pspec_clevel)]) 
        else:
           self.pspec_clevel = 3
        
        if 'pspec_iter' in kwargs:
           self.pspec_iter = kwargs['pspec_iter']
           self.para.append(['pspec_iter',str(self.pspec_iter)]) 
        else:
           self.pspec_iter = 150
           
        if 'map_alpha' in kwargs:
           self.map_alpha = kwargs['map_alpha']
           self.para.append(['map_alpha',str(self.map_alpha)]) 
        else:
           self.map_alpha = 1e-4
           
        if 'map_tol' in kwargs:
           self.map_tol = kwargs['map_tol']
           self.para.append(['map_tol',str(self.map_tol)]) 
        else:
           self.map_tol = 1e-5
           
        if 'map_clevel' in kwargs:
           self.map_clevel = kwargs['map_clevel']
           self.para.append(['map_clevel',str(self.map_clevel)]) 
        else:
           self.map_clevel = 3
           
        if 'map_iter' in kwargs:
           self.map_iter = kwargs['map_iter']
           self.para.append(['map_iter',str(self.map_iter)]) 
        else:
           self.map_iter = 100
           
        if 'ncpu' in kwargs:
           self.ncpu = kwargs['ncpu']
           self.para.append(['ncpu',str(self.ncpu )]) 
        else:
           self.ncpu = 2
        
        if 'nrun' in kwargs:
           self.nrun = kwargs['nrun']
           self.para.append(['nrun',str(self.nrun)]) 
        else:
           self.nrun = 8

    def printparameters(self,place):
     
        f = open(place+'/numparameters.txt','w')
        for name in self.para:
            f.write('{0:15}: {1:15}'.format(name[0],name[1]) + '\n')

class simparameters(object):
    
    def __init__(self, params, kwargs):
        
        self.para = []
         
        if 'simpix' in kwargs:
            self.simpix = kwargs['simpix']
            self.para.append(['simpix',str(self.simpix)]) 
        else: 
            self.simpix = 100
            
        if 'nsources' in kwargs:
            self.nsources = kwargs['nsources']
            self.para.append(['nsources',str(self.nsources)]) 
        else: 
            self.nsources = 50
            
        if 'pfactor' in kwargs:
            self.pfactor = kwargs['pfactor']
            self.para.append(['pfactor',str(self.pfactor)]) 
        else: 
            self.pfactor = 5.
            
        if 'signal_seed' in kwargs:
           self.signal_seed = kwargs['signal_seed']
           self.para.append(['signal_seed',str(self.signal_seed)]) 
        else:
           self.signal_seed = 454810740
           
        if 'p0_sim' in kwargs:
           self.p0_sim = kwargs['p0_sim']
           self.para.append(['p0_sim',str(self.p0_sim)]) 
        else:
           self.p0_sim = 9.7e-18   
           
        if 'k0' in kwargs:
           self.k0 = kwargs['k0']
           self.para.append(['k0',str(self.k0)]) 
        else:
           self.k0 = 19099
           
        if 'sigalpha' in kwargs:
           self.sigalpha = kwargs['sigalpha']
           self.para.append(['sigalpha',str(self.sigalpha)]) 
        else:
           self.sigalpha = 2.
               
        if 'noise_seed' in kwargs:
           self.noise_seed = kwargs['noise_seed']
           self.para.append(['noise_seed',str(self.noise_seed)]) 
        else:
           self.noise_seed = 3127312
           
        if 'sigma' in kwargs:
           self.sigma = kwargs['sigma']
           self.para.append(['sigma',str(self.sigma)]) 
        else:
           self.sigma = 1e-2
           
        if 'offset' in kwargs:
           self.offset = kwargs['offset']
           self.para.append(['offset',str(self.offset)]) 
        else:
           self.offset = 0.
           
        if 'compact' in kwargs:
           self.compact = kwargs['compact']
           self.para.append(['compact',str(self.compact)]) 
        else:
           self.compact = False

    def printparameters(self,place):
     
        f = open(place+'/simparameters.txt','w')
        for name in self.para:
            f.write('{0:15}: {1:15}'.format(name[0],name[1]) + '\n')

def make_dirtyimage(params, logger):
    
#    im.open(params.ms)
#    im.defineimage(cellx = str(2 * params.cellsize) + 'rad',celly = \
#        str(2 * params.cellsize) + 'rad', nx = params.imsize,\
#        ny = params.imsize)
#    im.clean(image='di', niter=0)
#    im.close()
    casa.clean(vis = params.ms,imagename = 'general_output/di',cell = \
        str(params.cellsize) + 'rad', imsize = params.imsize, \
        niter = 0)
    ia.open('general_output/di.image')
    imageinfo = ia.summary('general_output/di.image')
    
    norm = imageinfo['incr'][1]/np.pi*180*3600  #cellsize converted to arcsec
    
    beamdict = ia.restoringbeam()
    major = beamdict['major']['value'] / norm
    minor = beamdict['minor']['value'] / norm
    np.save('beamarea',1.13 * major * minor)
    
    di = ia.getchunk().reshape(imageinfo['shape'][0],imageinfo['shape'][1])\
        / (1.13 * major * minor)
    
    ia.close()
    
    call(["rm", "-rf", "di*"])
    
    return di
                

def callbackfunc(x, i):
    
    if i%gcallback == 0:
        print 'Callback at iteration' + str(i)
        
        if gsave:
           pl.figure()
           pl.imshow(np.exp(x))
           pl.colorbar()
           pl.title('Iteration' + str(i))
           pl.savefig("last_iterations/" + 'iteration'+str(i))
           np.save("last_iterations/" + 'iteration' + str(i),x)

def callbackfunc_u(x, i):
    
    if i%gcallback == 0:
        print 'Callback at iteration' + str(i) +' of the point source reconstruction'
        
        if gsave:
           pl.figure()
           pl.imshow(np.exp(x))
           pl.colorbar()
           pl.title('Iteration' + str(i)+'_u')
           pl.savefig("last_iterations/" + 'iteration'+str(i)+'_expu')
           np.save("last_iterations/" + 'iteration' + str(i)+'_expu',x)
               
def callbackfunc_m(x, i):
    
    if i%gcallback == 0:
        print 'Callback at iteration' + str(i) +' of the point source reconstruction'
        
        if gsave:
           pl.figure()
           pl.imshow(np.exp(x))
           pl.colorbar()
           pl.title('Iteration' + str(i)+'_m')
           pl.savefig("last_iterations/" + 'iteration'+str(i)+'_expm')
           np.save("last_iterations/" + 'iteration' + str(i)+'_expm',x)

def simulate(params, simparams, logger):
    """
    Setup for the simulated signal.
    """

    logger.header2("Simulating signal and data using provided UV-coverage.")

    u,v = read_data_from_ms(params.ms)[2:4]

    # wide-band imaging
    if params.freq == 'wideband':
        logger.message('Wideband imaging not yet implemented')
    # single-band imaging
    else:
        nspw,chan = params.freq[0], params.freq[1]
        u = u[nspw][chan]
        v = v[nspw][chan]
    
    
    d_space = point_space(len(u), datatype = np.complex128)
    s_space = rg_space(simparams.simpix, naxes=2, dist = params.cellsize, \
        zerocenter=True)
    k_space = s_space.get_codomain()
    kindex,rho_k,pindex,pundex = k_space.get_power_indices()
    
    #setting up signal power spectrum
    powspec_I = [simparams.p0_sim * (1. + (k / simparams.k0) ** 2) ** \
        (-simparams.sigalpha) for k in kindex]
    if params.save:      
        save_results(kindex,'simulated signal PS',"general_output/" + \
            params.save + '_ps_original', log = \
            'loglog', value2 = powspec_I)

    S = power_operator(k_space, spec=powspec_I)

    # extended signal
    np.random.seed(simparams.signal_seed)
    I = field(s_space, random="syn", spec=S.get_power()) + simparams.offset
    np.random.seed()    
    
    # compact signal
    if simparams.compact:
        Ip = np.zeros((simparams.simpix,simparams.simpix))
        for i in range(simparams.nsources):
               Ip[np.random.randint(0,high=simparams.simpix),\
               np.random.randint(0,high=simparams.simpix)] = \
               np.random.random() * simparams.pfactor * np.max(exp(I))  
        I += Ip           
   
    if params.save:      
        save_results(exp(I),'simulated signal',"general_output/" + \
            params.save + '_expsimI')
    
    
    if params.save:
        # maximum k-mode and resolution of data
        uvrange = np.array([np.sqrt(u[i]**2 + v[i]**2) for i in range(len(u))])
        dx_real_rad = (np.max(uvrange))**-1
        logger.message('resolution\n' + 'rad ' + str(dx_real_rad) + '\n' + \
            'asec ' + str(dx_real_rad/asec2rad))

        save_results(u,'UV',"general_output/" +  params.save + "_uvcov", \
            plotpar='o', value2 = v)

    # noise
    N = diagonal_operator(domain=d_space, diag=simparams.sigma**2)
    
    # response, no simulated primary beam
    A = 1.
    R = r.response(s_space, sym=False, imp=True, target=d_space, \
                       para=[u,v,A,False])
                 
    #Set up Noise
    np.random.seed(simparams.noise_seed)
    n = field(d_space, random="gau", var=N.diag(bare=True))
    # revert to unseeded randomness
    np.random.seed()  
    
    #plot Signal to noise
    sig = R(field(s_space, val = np.exp(I)))
    if params.save:
        save_results(abs(sig.val) / abs(n.val),'Signal to noise', \
            "general_output/" + params.save + '_StoN',log =\
            'semilog')

    d = R(exp(I)) + n
    
    # reset imsize settings for requested parameters
    s_space = rg_space(params.imsize, naxes=2, dist = params.cellsize, \
        zerocenter=True)
    R = r.response(s_space, sym=False, imp=True, target=d_space, \
                       para=[u,v,A,False])
    
    # dirty image for comparison
    di = R.adjoint_times(d) * s_space.vol[0] * s_space.vol[1]
    
    return d, N, R, di, d_space, s_space
    
    
def save_results(value,title,fname,log = None,value2 = None, \
    value3= None, plotpar = None, rho0 = 1., twoplot=False):
        
    rho0 = 1
    
    # produce plot and save it as png file   
    pl.figure()
    pl.title(title)
    if plotpar:
        if log == 'loglog' :
            pl.loglog(value,value2)
            if twoplot:
                pl.loglog(value,value3,plotpar)
        elif log == 'semilog':
            pl.semilogy(value)
            if twoplot:
                pl.semilogy(value3,plotpar)
        else :
            if len(np.shape(value)) > 1:
                pl.imshow(value/rho0)
                pl.colorbar()
            else:
                pl.plot(value,value2,plotpar)
                
            pl.savefig(fname + ".png")
    else:
        if log == 'loglog' :
            pl.loglog(value,value2)
            if twoplot:
                pl.loglog(value,value3)
        elif log == 'semilog':
            pl.semilogy(value)
            if twoplot:
                pl.semilogy(value3)
        else :
            if len(np.shape(value)) > 1:
                pl.imshow(value/rho0)
                pl.colorbar()
            else:
                pl.plot(value,value2)
        pl.savefig(fname + ".png")          
        
    pl.close
    
    # save data as npy-file
    if rho0 != 1.:
       np.save(fname,value/rho0)
    else:
       np.save(fname,value)


#*******************************************************************************
# Define truncatd exp and log functions for nifty fields to avoid NANs*********

def exp(x):

    if(isinstance(x,field)):
#        if(np.any(x.val>709)):
 #            print("** EXPSTROKE **")
        return field(x.domain,val=np.exp(np.minimum(709,x.val)),target=x.target)
        #return field(x.domain,val=np.exp(x.val),target=x.target)
    else:
#        if(np.any(x>709)):
#            print("** EXPSTROKE **")
        return np.exp(np.minimum(709,np.array(x)))
        #return np.exp(np.array(x))

def log(x,base=None):

    if(base is None):
        if(isinstance(x,field)):
#            if(np.any(x.val<1E-323)):
#                print("** LOGSTROKE **")
            return \
                field(x.domain,val=np.log(np.maximum(1E-323,x.val)),target=x.target)
            #return field(x.domain,val=np.log(x.val),target=x.target)
        else:
#            if(np.any(x<1E-323)):
#                print("** LOGSTROKE **")
            return np.log(np.array(np.maximum(1E-323,x)))
            #return np.log(np.array(x))

    base = np.array(base)
    if(np.all(base>0)):
        if(isinstance(x,field)):
#            if(np.any(x.val<1E-323)):
#                print("** LOGSTROKE **")
            return field(x.domain,val=np.log(np.maximum(1E-323,x.val))/np.log(base).astype(x.domain.datatype),target=x.target)
            #return field(x.domain,val=np.log(x.val)/np.log(base).astype(x.domain.datatype),target=x.target)
        else:
#            if(np.any(x<1E-323)):
#                print("** LOGSTROKE **")
            return np.log(np.array(np.maximum(1E-323,x)))/np.log(base)
    else:
        raise ValueError(about._errors.cstring("ERROR: invalid input basis."))
        

    
    
    
