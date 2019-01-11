###########
# IMPORTS #
###########

import numpy as py
import sncosmo
from pyifu import spectroscopy
SED_LIB_SOURCE = "../data/SED/templates/"

class SEDFitter( object ):
    
    def __init__(self, sdss_data, sdss_error=None, bands=["u","g","r","i","z"],  load_sedlib=True):
        """ provide dictionary 'u:.. g:.. }'"""
        self.photodata   = sdss_data
        self.photoerror  = sdss_error
        self.bands       = bands
        if load_sedlib:
            self.load_speclib()
        
    def load_speclib(self, used_met="008", clean = True, forcelbda=None):
        """ """
        files = glob(SED_LIB_SOURCE+'*_z%s*'%used_met) #one may change z008 to get other kinds of spec.
        self.specmodel_list = []

        for file in files :
            (lbda, flux) = np.asarray([l.split() for l in open(file).read().splitlines() if not l.startswith("#")],
                                      dtype="float").T
            spec1 = spectroscopy.get_spectrum(lbda, flux)
            if clean:
                spec1 = spec1.reshape(spec1.lbda[(spec1.lbda>3000) & (spec1.lbda<10000)])
                if forcelbda is not None:
                    spec1 = spec1.reshape(forcelbda)
                    
            self.specmodel_list.append(spec1)
            
            
    def load_sdssfilters(self,  bands=None):
        """ """
        if bands is None:
            bands = self.bands
            
        self.bandpasses = {}
        for b in bands:
            self.bandpasses[b]=sncosmo.get_bandpass('sdss'+b)
        
            
    def get_model_spec(self, weights, update=False):
        """ linear combination of spectra from specmodel_list 
        returns modelspectrum
        """
        list_model_flux = np.asarray([s_.data for s_ in self.specmodel_list])
        modelflux = np.dot(weights, list_model_flux)
        specmodel = spectroscopy.get_spectrum(self.reflbda, modelflux)
        if update:
            self.current_modelspectrum = specmodel
            
        return specmodel
    
    
    def get_model_photometry(self, weights, bands=None, update=False):
        """ """
        if bands is None:
            bands = self.bands
        if not hasattr(self, "bandpasses"):
            self.load_sdssfilters(bands=bands)
            
        modelspec =  self.get_model_spec(weights,update=update)
        photoband = []
        for b in bands:
            lbda_spec = modelspec.lbda
            flux_spec = modelspec.data
            photoband.append(spectroscopy.synthesize_photometry(lbda_spec, flux_spec, 
                                                            lbda_spec, self.bandpasses[b](lbda_spec)))
        if update:
            self.current_modelphoto = photoband
            
        return np.asarray(photoband)
    
    ##############
    # CHI2 & FIT #
    ##############
    
    def get_chi2(self, weights):
        """ """
        model = self.get_model_photometry(weights, self.bands)
        self.res = self.photodata-model
        if self.photoerror is not None:
            return np.sum( self.res**2/self.photoerror**2 )
        return  np.sum( self.res**2 )

    def fit(self, weight_guess=None):
        """ """
        from scipy.optimize import minimize
        if weight_guess is None:
            weight_guess = np.ones(self.nmodels)
        bounds = [[0,None] for i in range(self.nmodels)]
        self.fitresults = minimize(self.get_chi2, weight_guess, bounds=bounds)
        return self.fitresults
    
    ###########
    #  PLOTS  #
    ###########
    
    def show(self, model=True):
        """ """
        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        
        ax.scatter(self.band_lbda, self.photodata, marker="o",s=80, zorder=9)
        if model and hasattr(self, "fitresults"):
            modelspec = self.get_model_spec(self.fitresults["x"] )
            modelphoto =  self.get_model_photometry(self.fitresults["x"] )
            modelspec.show(ax=ax, color="C1", zorder=5)
            ax.scatter(self.band_lbda, modelphoto, marker="o",s=80, facecolors="w", edgecolors="C2", lw=2, zorder=8 )
            
    def show_models(self):
        """ """
        import matplotlib.pyplot as plt
        if not hasattr(self, "specmodel_list"):
            raise AttributeError("please first run load_speclib() ")
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        for i,s_ in enumerate(self.specmodel_list):
            s_.show(ax=ax, color=plt.cm.viridis(i/self.nmodels))
        
    ##############
    # PROPERTIES #
    ##############
    
    @property
    def nmodels(self):
        """ """
        if not hasattr(self, "specmodel_list"):
            raise AttributeError("please first run load_speclib() ")
        return len(self.specmodel_list)
    
    @property
    def reflbda(self, indexref=0):
        """ """
        return self.specmodel_list[indexref].lbda
    
    @property
    def band_lbda(self):
        """ """
        if not hasattr(self, "bandpasses"):
            self.load_sdssfilters(bands=bands)
        return [self.bandpasses[b].wave_eff for b in self.bands]