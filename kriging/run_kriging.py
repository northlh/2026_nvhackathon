"""Module to process and run kriging inputs"""

import xarray as xr
import numpy as np
import rioxarray

from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.linear_model import LinearRegression


class KrigHackathon:
    """
    Class to run ordinary and regression kriging models

    Parameters
    ----------

    """

    def __init__(
        self,
        path_points: str,
        varname_points: str,
        varname_x: str,
        varname_y: str,
        varname_time: str,
        src_crs: str,
        dst_crs: str = "EPSG:5070",
        path_rasters: dict,
        varname_rasters: dict
    ):
    """
    Parameters
    ----------
    path_points: str
        Full path string to a .nc, expecting xr.Dataset
    varname_points: str
        Name of the point dataset varaible to interpolate
    varname_x: str
        Name of the x coordinate in the point dataset
    varname_y: str
        Name of the y coordinate in the point dataset
    src_crs: str | None
        EPSG string of the point coordinate reference system
    dst_crs: str
        EPSG string of the destination CRS
    path_rasters: dict
        Dictionary of the static feature filepaths. Expecting .nc
    varname_rasters: dict
        Dictionary of the varnames
    """
    
    self._path_points = path_points
    self._varname_points = varname_points
    self._varname_x = varname_x
    self._varname_y = varname_y
    self._varname_time = varname_time
    self._src_crs = src_crs
    self._dst_crs = 
    
    self._path_rasters = path_rasters
    self._varname_rasters = varname_rasters
    
    self.ds_points = xr.open_dataset(path_points)
    self.arr_points =  self.ds_points[self._varname_points].values

    def reproj_points(self
    ):
        ds = self.ds_points

        # TODO: try to pull crs if available with rxr, must be valid
        # if point_crs is None:
        #     try(point_crs = ds.rio.crs)

        transformer = Transformer.from_crs(
            self.src_crs,
            self.dst_crs,
            always_xy = True)

        # must convert to np arrays
        src_x, src_y = ds[self._varname_x].values, ds[self._varname_y].values

        # TODO: assert x and y as 1D, same length

        dst_x, dst_y = transformer.tranform(src_x, src_y)

        return(dst_x, dst_y)


    # TODO: make it so temporal aggregations are more flexible
    def get_kriging_arrays(self):
        """
        Aggregate point variables across time

        Return
        ------
        x: np.ndarray
            Valid x points in projected space
        y: np.ndarray
            Valid y points in projected space
        z: np.ndarray
            Valid points to interpolate in projected space
        """

        arr_points_avg = np.nanmean(self.arr_points, axis=self._varname_time)
        
        dst_x, dst_y = self.reproj_points()

        valid = np.isfinite(err_avg) & np.isfinite(albers_x) & np.isfinite(albers_y)
        
        x = dst_x[valid]
        y = dst_y[valid]
        z = arr_points_avg[valid]

        assert x.shape == y.shape == z.shape

        return(x, y, z)

    # TODO: maybe not make this just a wrapper? Support more than defaults?
    def run_ok(self):
        """
        Run basic kriging model

        Return
        ------
        ok: OridinaryKriging
            Model object
        """

        x, y, z = self.get_kriging_arrays()
        
        ok = OrdinaryKriging(
            x,
            y,
            z,
            variogram_model="exponential",
            enable_plotting=False,
        )

        z_hat, _ = ok.execute("points", x, y) # execute model at points
        assert np.allclose(z_hat, z) is True  # should be True (within numerical tolerance)
        
        return ok

    # TODO: assert grid must match the self._dst_crs
    def interpolate_grid(self, x_vect, y_vect):
        """
        Run kriging model on predefined grid

        Grid must match the self._dst_crs
        ***WARNING: VERY MEMORY INTENSE ON LARGE GRIDS***
        Return
        ------
        z_grid: np.ndarray
            Interpolated z values
        z_var: np.ndarray
            Covariance matrix
        """

        z_grid, z_var = ok.execute(
            "grid",
            x_vect,
            y_vect,
        )

        return(z_grid, z_var)

    
    

    def reproj_rasters():

        

        


def reproj_rasters():


def linear_regress():


def points_from_raster():

