#! /usr/bin/env python
#

""" Tools to take an astrobject insturment and create a slice of it """

import numpy as np

from shapely import geometry
from shapely import vectorized

from astrobject import instruments

#
# Image Slicer
# 

# 
class ImageSlicer( object ):
    """ From an astrobject instrument do a slice """
    
    def __init__(self,  instrument, cube=None):
        """  """
        self.instrument    = instrument
        self.photo_polygon = PhotoPolygon(self.instrument.data, self.instrument.var)
        if cube is not None:
            self.set_cube()
        
    def set_cube(self, cube, target_loc=[0,0], 
                 spaxel_in_arcsec=0.75, cube_rotation=0,
                 setup_slices=False, **kwargs):
        """ """
        self.cube       = cube
        self.cube_slice = self.cube.get_slice(lbda_trans=[self.instrument.bandpass.wave,
                                                          self.instrument.bandpass.trans], 
                           slice_object=True)
        
        self._set_cube_prop_(target_loc=target_loc, spaxel_in_arcsec=spaxel_in_arcsec,
                             cube_rotation=cube_rotation)
        
        if setup_slices:
            if not self.instrument.has_target():
                raise AttributeError("Cannot setup the slice (setup_pixel_slice=True) because there is no attached target")
            self.setup_slices(**kwargs)
            
    def _set_cube_prop_(self, target_loc=None, spaxel_in_arcsec=None, cube_rotation=None):
        """ 
        Parameters
        ----------
        cube_rotation: [float] -optional-
            Trigo direction Rotation angle [in radian] with respect to north
        
        """
        if target_loc is not None:
            self._cube_target_loc  = np.asarray(target_loc)
        if spaxel_in_arcsec is not None:
            self._cube_spaxel_arcsec = spaxel_in_arcsec
        if cube_rotation is not None:
            self._cube_rotation = cube_rotation

    def get_centroid_from_cube(self, cube=None, correct_for_rotation=True, 
                               get_vertices=False, **kwargs):
        """ """
        if cube is not None:
            self.set_cube(cube, **kwargs)
        if not hasattr(self, "cube"):
            raise AttributeError("No cube set, please call self.set_cube")
        
        spaxels_xy = np.asarray(self.cube.index_to_xy(self.cube.indexes)) - self._cube_target_loc
        
        if correct_for_rotation:
            yrot = self.instrument.wcs.get_rotations()[-1] + self._cube_rotation
            rotmat = np.asarray([[np.cos(yrot), -np.sin(yrot)],[np.sin(yrot), np.cos(yrot)]])
            spaxels_xy = np.dot(rotmat, spaxels_xy.T).T
            vertices = np.dot(rotmat, self.cube.spaxel_vertices.T).T*self.spaxel_unit_to_pixel
        else:
            vertices = self.cube.spaxel_vertices*self.spaxel_unit_to_pixel
            
        coords = spaxels_xy * self.spaxel_unit_to_pixel +\
               self.instrument.coords_to_pixel(self.target.ra,self.target.dec)
            
        return coords if not get_vertices else (coords, vertices)
        
    def get_photometry_in_vertices(self, pixel_vertices, use_subpixelization=False):
        """ measure the photometry within a given vertice.
        This is calling get_photometry from self.photo_polygon 
        
        Vertices coordinates must be in image pixel
        """
        return self.photo_polygon.get_photometry(pixel_vertices,use_subpixelization=use_subpixelization)
        
    def setup_slices(self, indexes=None, **kwargs):
        """ """
        from pyifu import spectroscopy
        
        if indexes is None:
            indexes = self.cube.indexes
        # Spaxels shapes and centroids [in pixels]
        centroids, shape_vertices   = self.get_centroid_from_cube(get_vertices=True, **kwargs)
        default_data = np.ones(len(centroids))*np.NaN
        # -> Defining the pixel prejection of the Slice
        self.pixel_slice  = spectroscopy.get_slice(default_data, centroids, 
                                                shape_vertices,  variance = None, 
                                                indexes=indexes)
        # -> Its corresponding slices "a la SEDM"
        self.spaxel_slice = spectroscopy.get_slice( default_data, 
                                                    np.asarray(self.cube.index_to_xy(self.cube.indexes)), 
                                                    self.cube.spaxel_vertices,  variance = None, 
                                                    indexes=indexes)
        # -> Its associate background "a la SEDM"        
        self.bkgd_slice   = spectroscopy.get_slice( np.zeros(self.spaxel_slice.nspaxels), 
                                                    np.asarray(self.cube.index_to_xy(self.cube.indexes)), 
                                                    self.cube.spaxel_vertices,  variance = None, 
                                                    indexes=indexes)

        # -> And the residual slice ( cube_slice - (spaxel_slice+bkgd_slice) )
        self.residual_slice = spectroscopy.get_slice( default_data, 
                                                    np.asarray(self.cube.index_to_xy(self.cube.indexes)), 
                                                    self.cube.spaxel_vertices,  variance = None, 
                                                    indexes=indexes)
        
        # => At this point, the just created Slices are empty.
        #    You need to run `fillup_slices()` to fill them up.
        
    def fillup_slices(self, indexes=None, use_subpixelization=False,
                          update_residual=True):
        """ Measure the """
        if not hasattr(self, "pixel_slice"):
            raise AttributeError("No slice setup. run setup_slice() first.")
        
        used_indexes = self.pixel_slice.indexes if indexes is None else indexes
        data, var = self.get_photometry_in_vertices( self.pixel_slice.get_index_vertices(used_indexes), use_subpixelization=use_subpixelization)
        self.pixel_slice.set_data(data, variance=var)
        self.spaxel_slice.set_data(data, variance=var)
        if update_residual:
            self.measure_residual()

    def measure_residual(self):
        """ fillup the residual slice = cube_slice - (spaxel_slice+bkgd_slice) """
        resdata = self.cube_slice.data - (self.spaxel_slice.data + self.bkgd_slice.data)
        if self.cube_slice.has_variance() and self.spaxel_slice.has_variance():
            resvar = self.cube_slice.variance + self.spaxel_slice.variance
        else:
            resvar = None

        self.residual_slice.set_data(resdata, variance=resvar)
        
    # ================ #
    #  Scale           #
    # ================ #
    def match_scale(self):
        """ """
        from scipy.optimize import minimize
        #
        # Only work at the data level to speed things up
        # minimize works better for number close to 1 (i.e. no 1e-16)
        #
        # Step 1: Get the single 1D array (data and model and background)
        norm_data  = self.cube_slice.rawdata.mean()
        norm_model = self.spaxel_slice.rawdata.mean()
        data_      = self.cube_slice.data   / norm_data
        var_       = 1 if not self.cube_slice.has_variance() else self.cube_slice.variance / norm_data**2
        model_     = self.spaxel_slice.data / norm_model
        bkgd_      = np.ones( len(model_) ) # No structure
        def _get_res_(amplitude, background):
            """ """
            return data_ - (model_*amplitude + bkgd_*background)
        
        def _chi2_( parameters ):
            """ """
            return np.sum(_get_res_(*parameters)**2/var_**2)
        
        self._rawres = minimize( _chi2_, [1,0] )
        self.residual_slice.set_data( _get_res_(*self._rawres["x"]), variance=var_)
        return self._rawres
        
        

        
    # ------ #
    # PLOT   #
    # ------ #
    def show(self, show_slice=True, target_color="C1",
                 slice_alpha=0.1,
                 slice_edge_alpha=0.8, 
                 slice_ec="0.5", slice_edgecolor=None, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        from matplotlib import patches
        fig = mpl.figure(figsize=[9,3])
        aximg   = fig.add_axes([0.05,0.1,0.25,0.8])
        axcube  = fig.add_axes([0.375,0.1,0.25,0.8])
        axmodel = fig.add_axes([0.7 ,0.1,0.25,0.8])
        _ = self.instrument.show(ax=aximg, zorder=3,
                                proptarget=dict(marker="x", mfc="None", mec=target_color, mew=1),
                                **{**dict(logscale=False, zoom=50, zoomon="target", vmax="99.8", vmin="0.2"), **kwargs})
        
        # Cube Slice        
        self.cube_slice.show(ax=axcube, vmax="99.5", show_colorbar=False)
        if hasattr(self, "_cube_target_loc") and self._cube_target_loc is not None:
            axcube.scatter(*self._cube_target_loc, marker="x", color=target_color, s=80, zorder=9)
        
        # Show slices
        if show_slice and hasattr(self, "pixel_slice"):
            if slice_edgecolor is None:
                slice_edgecolor = slice_ec
                
            self.pixel_slice.show(toshow=None, 
                                  ec=slice_ec, alpha=slice_alpha,
                            ax=aximg, show=False, 
                            zorder=5)
            self.pixel_slice.display_contours(ax=aximg, buffer=0.01,
                                            facecolor="None", edgecolor=slice_edgecolor,
                                            zorder=6, autoscale=False, alpha=slice_edge_alpha)
            
            self.spaxel_slice.show(ax=axmodel, show_colorbar=False)
            if hasattr(self, "_cube_target_loc") and self._cube_target_loc is not None:
                axmodel.scatter(*self._cube_target_loc, marker="x", color=target_color, s=80, zorder=9)


    def show_cube(self, toshow="data", cmap=None, vmin="0.5", vmax="99.5",
              filter_color="C0",filter_alpha=0.1, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        fig = mpl.figure(figsize=[10,3.5])
        axspec = fig.add_axes([0.10,0.15,0.5,0.75])
        axim   = fig.add_axes([0.65,0.15,0.26,0.75])
        axspec.set_xlabel(r"Wavelength", fontsize="large")
        axspec.set_ylabel(r"Flux", fontsize="large")

        slice_ = self.cube.get_slice()
        self.cube_slice.show( toshow=toshow,ax=axim, cmap=cmap, vmin=vmin, vmax=vmax, show_colorbar=False,**kwargs)
        self.cube._display_spec_(axspec, toshow,  **kwargs)
        # - Filter
        axfilt = axspec.twinx()
        axfilt.fill_between(self.instrument.bandpass.wave, self.instrument.bandpass.trans,
                            color=filter_color, alpha=filter_alpha)
        axfilt.text(self.instrument.bandpass.wave_eff, 0, r"%s"%(self.instrument.bandpass.name),
                ha="center", va="bottom", color=filter_color, alpha=np.min([filter_alpha*2,1]))
        # Fancy
        axfilt.set_ylim(0,1)
        axfilt.set_yticks([])
        return {"ax":[axim, axspec, axfilt],"fig":fig}
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def target(self):
        """ """
        return self.instrument.target
    
    def set_target(self, target):
        """ """
        self.instrument.set_target(target)
        
    @property
    def spaxel_unit_to_pixel(self):
        """ """
        return self._cube_spaxel_arcsec * self.instrument.units_to_pixels('arcsec').value


##########################
#                        #
#   INTERNAL TOOLS       #
#                        #
##########################
#
# Subpixelization methods
#
# = Most accurate
UNIT_SQUATE = np.asarray([[0,0],[0,1],[1,1],[1,0]])
def frac_pixel_within_vertices(pixels, vertices):
    """ Most accurate weighting method. """
    polygon = geometry.Polygon(vertices)
    pixel_square = np.asarray([UNIT_SQUATE+p_ for p_ in pixels])
    corners_in   = vectorized.contains( polygon, *pixel_square.reshape(len(pixel_square)*4,2).T
                                       ).reshape(len(pixel_square),4)
    weights = []    
    for pixel_s, corner_in in zip(pixel_square, corners_in):
        if np.all(corner_in):
           weights.append(1)
        elif np.any(corner_in):
            weights.append(polygon.intersection(geometry.Polygon(pixel_s)).area)
        else:
            weights.append(0)
            
    return weights

# = Slightly faster
SUBPIXELS  = {precision: np.mgrid[0:1+1/precision:1/(precision-1),0:1+1/precision:1/(precision-1)].T.reshape(precision**2, 2)
                for precision in np.arange(3,20)}
    
def subfrac_pixel_within_vertices(pixels, vertices, subpixels):
    """ Most accurate weighting method. """
    polygon = geometry.Polygon(vertices)
    pixel_splitted = np.asarray([p_+SUBPIXELS[subpixels] for p_ in pixels])
    subpixels_in   = vectorized.contains( polygon, *pixel_splitted.reshape(len(pixel_splitted)*(subpixels**2),2).T
                                       ).reshape(len(pixel_splitted), subpixels**2)
    
    return np.sum(subpixels_in.T, axis=0)/subpixels**2



def get_pixels_to_consider(vertices, buffer=1):
    """ """
    (xmin,ymin),(xmax, ymax) = np.percentile(vertices, [0,100], axis=0)
    pixels_ = np.mgrid[xmin-buffer:xmax+buffer, ymin-buffer:ymax+buffer].T
    init_shape = np.shape(pixels_)[:2]
    return np.asarray(pixels_.reshape(init_shape[0]*init_shape[1], 2), dtype="int")

######
# Multiprocessing
def _vertices_to_pixels_and_weights(vert_):
    """ """
    if np.any(np.isnan(vert_)):
        return [],np.NaN 
    pixels  = get_pixels_to_consider(vert_)
    weights = frac_pixel_within_vertices(pixels, vert_)
    return pixels, weights

def _vertices_to_pixels_and_weights_from_subpixels(vert_, subpixels=7):
    """ """
    if np.any(np.isnan(vert_)):
        return [],np.NaN 
    pixels  = get_pixels_to_consider(vert_)
    weights = subfrac_pixel_within_vertices(pixels, vert_, subpixels=subpixels)
    return pixels, weights

######

def get_slice_photometry(data, vertices, variance=None, use_subpixelization=True,
                         notebook=True, ncore=None, verbose=False):
    """ """
    
    def photometrize(data_, pixels_, weights_):
        """ """
        return np.nansum( weights_ * np.asarray([data_[tuple(p_)[::-1]] for p_ in pixels_]) )

    multi_photo = len(np.shape(vertices))==3

    func_vert_to_pixel_and_weight = _vertices_to_pixels_and_weights_from_subpixels if use_subpixelization else\
          _vertices_to_pixels_and_weights
    #
    # Step 1, measure the weight of pixels concerned by the vertices
    #
    if not multi_photo:
        pixels, weights = func_vert_to_pixel_and_weight(vertices)
    else:
        import multiprocessing
        if ncore is None:
            if multiprocessing.cpu_count()>8:
                ncore = multiprocessing.cpu_count() - 2
            else:
                ncore = multiprocessing.cpu_count() - 1
                
            if ncore==0:
                ncore = 1

        nruns = np.shape(vertices)[0]
        if verbose:
            print(" Multiprocessing ".center(30,"-"))
            print(" %d measurements to do "%nruns)
            print(" running on %d (requested) cores "%ncore)

        p = multiprocessing.Pool(ncore)
        pixels, weights = [],[]
        for j, result in enumerate( p.map(func_vert_to_pixel_and_weight, vertices) ):
            pixels.append(result[0])
            weights.append(result[1])
        # - end multi processing
    #
    # Step 2, Given the weights and the pixels, get the weighted sums (data and variance)
    #
    if not multi_photo:
        pdata  = photometrize(data, pixels, weights)
        pvar   = photometrize(variance, pixels, weights) if variance is not None else None
    else:
        pdata  = np.asarray([photometrize(data, pxls, wghts) for pxls, wghts in zip(pixels, weights)])
        pvar   = np.asarray([photometrize(variance, pxls, wghts) for pxls, wghts in zip(pixels, weights)]) if variance is not None else None
        
    return pdata, pvar

#
# Synthesize photometry in any polygon
# 
class PhotoPolygon( object ):
    """ """
    def __init__(self, data, variance=None, poly_vertices=None):
        """ """
        self.data = data
        self.variance = variance
        
    # ------ #
    # MAIN   #
    # ------ #
    def get_photometry(self, vertices=None, set_poly=False, **kwargs):
        """ Get the sum (np.nansum) of the product between self.data and the weightmask 
        derived from the given vertices (or already known one)"""
        if vertices is not None:
            if set_poly:
                self.set_polygon(vertices, update_mask=True)
        else:
            vertices = self.polyvertices

        return get_slice_photometry(self.data, vertices, variance= self.variance, **kwargs)

    # ------ #
    # CHECK  #
    # ------ #
    
    def set_polygon(self, vertices, update_mask=False):
        """ """
        if len(np.shape(vertices))==3:
            # Multi cases
            self.polygon  = geometry.MultiPolygon([geometry.Polygon(v_) for v_ in vertices])
            update_mask = False
        else:
            self.polygon  = geometry.Polygon(vertices)
            
        self.polymask = None
        if update_mask:
            _ = self.get_polygon_mask(True)

    def get_pixels_in(self, buffer=1):
        """ """
        return get_pixels_to_consider(self.polyvertices, buffer=buffer)
    
    def get_polygon_mask(self, update=False):
        """ """
        mask = np.zeros(self.data.shape)
        
        unit_square  = np.asarray([[0,0],[0,1],[1,1],[1,0]])
        pixels       = self.get_pixels_in()
        pixel_square = np.asarray([unit_square+p_ for p_ in pixels])
        corners_in   = vectorized.contains( self.polygon, *pixel_square.reshape(len(pixel_square)*4,2).T
                                       ).reshape(len(pixel_square),4)
        
        for pixel_, pixel_s, corner_in in zip(pixels,pixel_square, corners_in):
            if np.all(corner_in):
                mask[tuple(pixel_)[::-1]] = 1
            elif np.any(corner_in):
                mask[tuple(pixel_)[::-1]] = self.polygon.intersection(geometry.Polygon(pixel_s)).area
            else:
                mask[tuple(pixel_)[::-1]] = 0
            
        if update:
            self.polymask = mask
            
        return mask
    
    # ------ #
    # PLOT   #
    # ------ #
    def show(self):
        """ """
        import matplotlib.pyplot as mpl
        fig   = mpl.figure()
        aximg = fig.add_axes([0.1,0.1,0.37,0.7])
        axmask= fig.add_axes([0.6,0.1,0.37,0.7])
        
        prop_ = dict(origin="lower")
        aximg.imshow(self.data, **prop_)
        axmask.imshow(self.get_polygon_mask(), **prop_)
        
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def polyvertices(self):
        """ """
        return self.polygon.exterior.xy.T
    
    @property
    def polyextrema(self):
        """ if within a rectangle, the lower-left and upper-right corners """
        return np.percentile(self.polyvertices, [0,100], axis=1)
    
    @property
    def polymask_inclusive(self):
        """ The polymask where all non-zero pixels are set to 1 """
        return np.asarray(self.polymask>0, dtype="int")
    
    @property
    def polymask_inclusive(self):
        """ The polymask where all non-one pixels are set to 0 """
        return np.asarray(self.polymask==1, dtype="int")
    
    def has_variance(self):
        """ has the variance been set ?"""
        return self.variance is not None
