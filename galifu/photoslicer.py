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
                 spaxel_in_arcsec=0.75, 
                 setup_slices=False, **kwargs):
        """ """
        self.cube = cube
        self._cube_target_loc  = np.asarray(target_loc)
        self._cube_spaxel_arcsec = spaxel_in_arcsec
        if setup_slices:
            if not self.instrument.has_target():
                raise AttributeError("Cannot setup the slice (setup_pixel_slice=True) because there is no attached target")
            self.setup_slices(**kwargs)
            
    def get_centroid_from_cube(self, cube=None, correct_for_rotation=True, 
                               get_vertices=False, **kwargs):
        """ """
        if cube is not None:
            self.set_cube(cube, **kwargs)
        if not hasattr(self, "cube"):
            raise AttributeError("No cube set, please call self.set_cube")
        
        spaxels_xy = np.asarray(self.cube.index_to_xy(self.cube.indexes))
        if correct_for_rotation:
            yrot = self.instrument.wcs.get_rotations()[-1]
            rotmat = np.asarray([[np.cos(yrot), -np.sin(yrot)],[np.sin(yrot), np.cos(yrot)]])
            spaxels_xy = np.dot(rotmat, spaxels_xy.T).T
            vertices = np.dot(rotmat, self.cube.spaxel_vertices.T).T*self.spaxel_unit_to_pixel
        else:
            vertices = self.cube.spaxel_vertices*self.spaxel_unit_to_pixel
            
        coords = (spaxels_xy - self._cube_target_loc
                                ) * self.spaxel_unit_to_pixel +\
                            self.instrument.coords_to_pixel(self.target.ra,self.target.dec)
            
        return coords if not get_vertices else (coords, vertices)
        
    def get_photometry_in_vertices(self, pixel_vertices):
        """ measure the photometry within a given vertice.
        This is calling get_photometry from self.photo_polygon 
        
        Vertices coordinates must be in image pixel
        """
        return self.photo_polygon.get_photometry(pixel_vertices)
        
    def setup_slices(self, indexes=None, **kwargs):
        """ """
        from pyifu import spectroscopy
        
        if indexes is None:
            indexes = self.cube.indexes
        
        centroids, shape_vertices   = self.get_centroid_from_cube(get_vertices=True, **kwargs)
        default_data = np.ones(len(centroids))*np.NaN
        self.pixel_slice  = spectroscopy.get_slice(default_data, centroids, 
                                                shape_vertices,  variance = None, 
                                                indexes=indexes)
        self.spaxel_slice = spectroscopy.get_slice(default_data, 
                                                    np.asarray(self.cube.index_to_xy(self.cube.indexes)), 
                                                    self.cube.spaxel_vertices,  variance = None, 
                                                    indexes=indexes)
        
    def fillup_slices(self, indexes=None):
        """ """
        if not hasattr(self, "pixel_slice"):
            raise AttributeError("No slice setup. run setup_slice() first.")
        
        used_indexes = self.pixel_slice.indexes if indexes is None else indexes
        data, var = np.asarray([self.get_photometry_in_vertices(index_ver_)
                                for index_ver_ in self.pixel_slice.get_index_vertices(used_indexes)]).T
        self.pixel_slice.set_data(data, variance=var)
        self.spaxel_slice.set_data(data, variance=var)
        
        
    # ------ #
    # PLOT   #
    # ------ #
    def show(self, show_slice=True, slice_alpha=0.8, 
             slice_ec="0.5", **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        from matplotlib import patches
        fig = mpl.figure(figsize=[9,3])
        aximg   = fig.add_axes([0.05,0.1,0.25,0.8])
        axcube  = fig.add_axes([0.375,0.1,0.25,0.8])
        axmodel = fig.add_axes([0.7 ,0.1,0.25,0.8])
        
        _ = self.instrument.show(ax=aximg, zorder=3,**kwargs)
        # Show slices
        if show_slice and hasattr(self, "pixel_slice"):
            self.pixel_slice.show(toshow=None, 
                                  ec=slice_ec, alpha=slice_alpha,
                            ax=aximg, show=False, 
                            zorder=5)
            self.spaxel_slice.show(ax=axmodel, show_colorbar=False)
            
        self.cube._display_im_(axcube)
        
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
    # SETTER #
    # ------ #
    def set_polygon(self, vertices, update_mask=False):
        """ """
        self.polygon  = geometry.Polygon(vertices)
        self.polymask = None
        if update_mask:
            _ = self.get_polygon_mask(True)
        
    # ------ #
    # GETTER #
    # ------ #
    def get_photometry(self, vertices=None):
        """ Get the sum (np.nansum) of the product between self.data and the weightmask 
        derived from the given vertices (or already known one)"""
        if vertices is not None:
            if np.any(np.isnan(vertices)): # Nan if bad vertices
                return (np.NaN,np.NaN) if self.has_variance() else (np.NaN, None)
            self.set_polygon(vertices, update_mask=True)
        elif not hasattr(self, "polygon"):
            raise AttributeError("no known polygon. provide `vertices` or call set_polygon()")
        elif not hasattr(self, "polymask"):
            _ = self.get_polygon_mask(True)
            
        if not self.has_variance():
            return np.nansum(self.data * self.polymask), None
        
        return np.nansum(self.data * self.polymask), np.nansum(self.variance * self.polymask)
    
    def get_pixels_in(self, buffer=2):
        """ """
        (xmin,ymin),(xmax, ymax) = self.polyextrema
        pixels_ = np.mgrid[xmin-buffer:xmax+buffer, ymin-buffer:ymax+buffer].T
        init_shape = np.shape(pixels_)[:2]
        return np.asarray(pixels_.reshape(init_shape[0]*init_shape[1], 2), dtype="int")
        
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
        return self.polygon.exterior.xy
    
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
