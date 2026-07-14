from __future__ import annotations

from xorq.vendor.ibis.formats.pandas import PandasData


class BigQueryPandasData(PandasData):
    @classmethod
    def convert_GeoSpatial(cls, s, dtype, pandas_type):
        import geopandas as gpd
        import shapely as shp

        try:
            return gpd.GeoSeries.from_wkt(s)
        except shp.geos.GEOSException:
            return gpd.GeoSeries.from_wkb(s)

    convert_Point = convert_LineString = convert_Polygon = convert_MultiLineString = (
        convert_MultiPoint
    ) = convert_MultiPolygon = convert_GeoSpatial
