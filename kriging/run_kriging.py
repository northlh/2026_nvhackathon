"""Module to process and run kriging inputs"""

import xarray as xr
import numpy as np
import rioxarray
import rasterio
import warnings

from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.linear_model import LinearRegression


class KrigHackathon:
    """
    Class to run ordinary and regression kriging models
    """

    def __init__(
        self,
        path_points: str,
        varname_points: str,
        varname_x: str,
        varname_y: str,
        varname_time: str,
        src_crs: str,
        ProjWUS
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
            EPSG string of the point coordinate reference system.
            Could be pulled from data, access varies by data format. 
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
        
        self.ProjWUS = ProjWUS 
    
        self.ds_points = xr.open_dataset(path_points)
        self.arr_points =  self.ds_points[self._varname_points].values

    def reproj_points(self
    ):
        ds = self.ds_points

        # TODO: try to pull crs if available with rxr, must be valid
        # if point_crs is None:
        #     try(point_crs = ds.rio.crs)

        transformer = Transformer.from_crs(
            self._src_crs,
            self.ProjWUS._dst_crs,
            always_xy = True)

        # must convert to np arrays
        src_x, src_y = ds[self._varname_x].values, ds[self._varname_y].values

        # TODO: assert x and y as 1D, same length

        dst_x, dst_y = transformer.transform(src_x, src_y)

        # # Debug
        # print(np.nanmin(dst_x), np.nanmax(dst_x))
        # print(np.nanmin(dst_y), np.nanmax(dst_y))
        
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

        # # TODO: somehow make this work instead of hardcoding the axis idx        
        # arr_points_avg = np.nanmean(
        #     self.arr_points,
        #     axis=self.ds_points[self._varname_points].get_axis_num(self._varname_time))

        arr_points_avg = np.nanmean(
            self.arr_points,
            axis=0)
        msg = "Hardcoded threshold time position index as 0"
        warnings.warn(msg)        
        
        dst_x, dst_y = self.reproj_points()

        
        valid = (np.isfinite(arr_points_avg) & 
                 np.isfinite(dst_x) & 
                 np.isfinite(dst_y) &
                 (arr_points_avg > -10))
        msg = "Hardcoded threshold > -10 for ws_error"
        warnings.warn(msg)
        
        
        x = dst_x[valid]
        y = dst_y[valid]
        z = arr_points_avg[valid]

        assert x.shape == y.shape == z.shape

        return(x, y, z)

    # TODO: maybe not make this just a wrapper? Support more than defaults?

    # @staticmethod
    # def run_ok(self, x, y, z):
    #     """
    #     Run basic kriging model

    #     Return
    #     ------
    #     ok: OridinaryKriging
    #         Model object
    #     """

    #     #x, y, z = self.get_kriging_arrays()
        
    #     ok = OrdinaryKriging(
    #         x,
    #         y,
    #         z,
    #         variogram_model="exponential",
    #         enable_plotting=False,
    #     )

    #     z_hat, _ = ok.execute("points", x, y) # execute model at points
    #     assert np.allclose(z_hat, z) is True  # should be True
        
    #     return ok

    # TODO: assert grid must match the self._dst_crs
    def interpolate_grid(self, x, y, z):
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
        # Define kriging model with point arrays
        ok = OrdinaryKriging(
            x,
            y,
            z,
            variogram_model="exponential",
            enable_plotting=False,
        )

        z_hat, _ = ok.execute("points", x, y) # execute model at points
        assert np.allclose(z_hat, z) is True  # should be True

        # Execute the kriging model over the prediction grid
        z_grid, z_var = ok.execute(
            "grid",
            self.ProjWUS.x_vect,
            self.ProjWUS.y_vect,
        )

        return(z_grid, z_var)

    # TODO: could make this more robust to loop through multiple datasets
    # TODO: could make it pull the src_crs from the data, not always there tho
    # TODO: currently designed for .nc from Bias Correction work, no GeoTIFFs
    def reproj_rasters(
        self,
        ds_path,
        varname,
        src_crs
    ):
        """
        Reporjects a raster and samples it at kriging points

        Parameters
        ----------
        ds_path: str
            Full filepath of the raster, expecting .nc
        varname: str
            Name of the variable name in the dataset
        src_crs: str
            EPSG string of the source raster

        Return
        ------
        tuple(grid_out, points_out)
            grid_out: np.ndarray
                Raster data reprojected to the prediction grid.
            points_out: np.ndarray
                Point data extracted from the raster. 
        """
        kx, ky, _ = self.get_kriging_arrays()
        
        ds = xr.open_dataset(ds_path)
        ds = ds.rio.write_crs(src_crs)

        # Reproject
        ds_out = ds.rio.reproject(
            dst_crs=self.ProjWUS._dst_crs,
            transform=self.ProjWUS.transform,
            shape=(self.ProjWUS.height, self.ProjWUS.width),
            resampling=rasterio.enums.Resampling.nearest,  # fastest          
        )

        # Sample at kriging points
        vals = ds_out.sel(
            x=xr.DataArray(kx, dims="points"),
            y=xr.DataArray(ky, dims="points"),
            method="nearest", 
        )

        grid_out = ds_out[varname].values.flatten()
        points_out = vals[varname].values

        return(grid_out, points_out)

    @staticmethod
    # TODO: maybe move away from a list kwarg, kindof fragile. Prefer tuple. 
    def stack_raster_data(
        grid_out: np.ndarray | list,
        points_out: np.ndarray | list
    ):
        """
        Stack flattened grids and point vectors column wise.

        Needed when there are more than one features.
        Pass a list of outputs from the reproj_rasters method.

        Parameters
        ----------
        grid_out: np.ndarray | list
            Flattened grids of raster features stacked column-wise
        points_out: np.ndarray | list
            Points extracted from the rasters stacked column-wise

        Return
        ------
        tuple(predictor_grid, predictor points)
            predictor_grid: np.ndarray
                Spatially complete array of predictor vars for the regression
            predictor_points: np.ndarray
                Spatially sparse array of predictor variables for the regression        
        """
        predictor_grid = np.column_stack(
            grid_out
        )

        predictor_points = np.column_stack(
            points_out
        )

        return(predictor_grid, predictor_points)

        
    def run_regression(
        self,
        predictor_grid,
        predictor_points,
        z):
        """
        Run a linear regression on using predictors from rasters.


        Parameters
        ----------
        predictor_grid: np.ndarray
            Spatially complete array of predictor variables for the regression.
            From stack_raster_data or reproj_rasters method. 
        predictor_points: np.ndarray
            Spatially sparse array of predictor variables for the regression.
            From stack_raster_data or reproj_rasters method. 
        z: np.ndarray
            The predictand, derived from the get_kriging_arrays method or
            separately.

        Return
        ------
        tuple(z_hat_grid, z_hat_points)
            z_hat_grid: np.ndarray
                Spatially complete array of the predictand at the defined grid
            z_hat_points: np.ndarray
                Spatially sparse array of the predictand at the points       
        """
        # TODO: assert array shapes with nice error message
        reg = LinearRegression()
        reg.fit(predictor_points, z)
        
        z_hat_points = reg.predict(predictor_points)

        z_hat_grid = reg.predict(predictor_grid).reshape(
            self.ProjWUS.height, self.ProjWUS.width)

        return(z_hat_grid, z_hat_points)
        

