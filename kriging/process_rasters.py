"""Module adapted from fwx.data_process.dp_regrid_wus"""

import rasterio
import logging
import geopandas as gpd
import numpy as np

from shapely.geometry import box

# PARAMETERS OF CANONICAL GRID
DST_CRS = "EPSG:5070"
RES_X_M, RES_Y_M = 3000, 3000
#RES_X_M, RES_Y_M = 990, 990
ORIGIN_X, ORIGIN_Y = 0, 0
#LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = (-125, 30, -96, 50) #WUS
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = (-111, 37, -102, 45) #COWY
BBOX_CRS = "EPSG:4326"

logger = logging.getLogger("fwx")

def snap_to_grid(bounds, origin_x, origin_y, res_x, res_y):
    """
    Define bounds on even grid cell intervals
    
    :param bounds: Description
    :param origin_x: Description
    :param origin_y: Description
    :param res_x: Description
    :param res_y: Description
    """

    xmin, ymin, xmax, ymax = bounds
    xmin_s = origin_x + np.floor((xmin - origin_x) / res_x) * res_x
    xmax_s = origin_x + np.ceil((xmax - origin_x) / res_x) * res_x
    ymin_s = origin_y + np.floor((ymin - origin_y) / res_y) * res_y
    ymax_s = origin_y + np.ceil((ymax - origin_y) / res_y) * res_y
    return xmin_s, ymin_s, xmax_s, ymax_s


class ProjWUS:
    """
    Class to define grid parameters

    Parameters
    ----------
    dst_crs
    res_x_m
    res_y_m
    origin_x
    origin_y
    lon_min
    lat_min
    lon_max
    lat_max
    bbox_crs
    """

    def __init__(
        self,
        dst_crs: str = DST_CRS,
        res_x_m: float = RES_X_M,
        res_y_m: float = RES_Y_M,
        origin_x: float = ORIGIN_X,
        origin_y: float = ORIGIN_Y,
        lon_min: float = LON_MIN,
        lat_min: float = LAT_MIN,
        lon_max: float = LON_MAX,
        lat_max: float = LAT_MAX,
        bbox_crs: str = BBOX_CRS
    ):
        self._dst_crs = dst_crs
        self._res_x_m = res_x_m
        self._res_y_m = res_y_m
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._lon_min = lon_min
        self._lat_min = lat_min
        self._lon_max = lon_max
        self._lat_max = lat_max
        self._bbox_crs = bbox_crs

        self._frozen = True

    # Force immutability
    # def __setattr__(self, name, value):
    #     if hasattr(self, "_frozen") and name not in {"_frozen"}:
    #         raise AttributeError(
    #             f"{self.__class__.__name__} instances are immutable"
    #         )
    #     super().__setattr__(name, value)

    def __setattr__(self, name, value):
        if hasattr(self, "_frozen"):
            # Block changes to public attributes only
            if not name.startswith("_"):
                raise AttributeError(
                    f"{self.__class__.__name__} instances are immutable"
                )
        super().__setattr__(name, value)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"dst_crs={self._dst_crs!r},\n"
            f"resolution=({self._res_x_m}, {self._res_y_m}),\n"
            f"origin=({self._origin_x}, {self._origin_y}),\n"
            f"aoi_bbox=({self._lon_min}, {self._lat_min},"
            f"{self._lon_max}, {self._lat_max}),\n"
            f"shape=({self.height}, {self.width}),\n"
            f"tranform=({self.transform})\n"
            f")"
        )

    @property
    def aoi_bbox(self):
        """
        Return AOI bounding box in soure coordinates
        
        :param self: Description
        """
        if not hasattr(self, "_aoi_bbox"):
            self._aoi_bbox = gpd.GeoDataFrame(
                geometry=[box(self._lon_min, self._lat_min, 
                              self._lon_max, self._lat_max)],
                crs=self._bbox_crs
            )
        return self._aoi_bbox

    @property
    def aoi_dst(self):
        """
        Return AOI bounding box in destination coordinates
        
        :param self: Description
        """
        if not hasattr(self, "_aoi_dst"):
            self._aoi_dst = self.aoi_bbox.to_crs(self._dst_crs)
        return self._aoi_dst

    @property
    def aoi_bounds(self):
        """
        Return values of AOI bounds in destination coordinates
        
        :param self: Description
        """
        if not hasattr(self, "_aoi_bounds"):
            self._aoi_bounds = self.aoi_dst.total_bounds
        return self._aoi_bounds

    @property
    def snapped_bounds(self):
        """AOI bounds snapped to canonical grid."""
        if not hasattr(self, "_snapped_bounds"):
            self._snapped_bounds = snap_to_grid(
                bounds=self.aoi_bounds,
                origin_x=self._origin_x,
                origin_y=self._origin_y,
                res_x=self._res_x_m,
                res_y=self._res_y_m,
            )

        return self._snapped_bounds

    @property
    def width(self):
        """
        Return width of destination grid
        
        :param self: Description
        """
        if not hasattr(self, "_width"):
            xmin, _, xmax, _ = self.snapped_bounds
            self._width = int((xmax - xmin) / self._res_x_m)
        return self._width
    
    @property
    def height(self):
        """
        Return height of destination grid
        
        :param self: Description
        """
        if not hasattr(self, "_height"):
            _, ymin, _, ymax = self.snapped_bounds
            self._height = int((ymax - ymin) / self._res_y_m)
        return self._height

    @property
    def transform(self):
        """
        Return rasterio destination transform
        
        :param self: Description
        """
        if not hasattr(self, "_transform"):
            xmin, _, _, ymax = self.snapped_bounds
            self._transform = rasterio.transform.from_origin(
                west = xmin, north = ymax,
                xsize = self._res_x_m, ysize = self._res_y_m
            )
        return self._transform

    ##### NEW CODE
    @property
    def y_vect(self):
        """
        Return rasterio destination transform
        
        :param self: Description
        """
        if not hasattr(self, "_y_vect"):
            _, ymin, _, ymax = self.snapped_bounds
            dy = self._res_y_m
            self._y_vect = np.arange(
                ymin + dy / 2, # grid center
                ymax,
                dy)
        return self._y_vect


    @property
    def x_vect(self):
        """
        Return rasterio destination transform
        
        :param self: Description
        """
        if not hasattr(self, "_x_vect"):
            xmin, _, xmax,_ = self.snapped_bounds
            dx = self._res_x_m
            self._x_vect = np.arange(
                xmin + dx / 2, # grid center
                xmax,
                dx)
        return self._x_vect
