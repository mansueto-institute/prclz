from abc import ABC as AbstractBaseClass
from abc import ABCMeta, abstractclassmethod
from logging import info
from typing import Union, Optional

from shapely.geometry import (MultiLineString, MultiPolygon, Polygon, asShape,
                              mapping)
from shapely.ops import polygonize


class BlockExtractionMethod(AbstractBaseClass, metaclass=ABCMeta):
    @abstractclassmethod
    def extract(self, region: Union[Polygon, MultiPolygon], linestrings: MultiLineString) -> MultiPolygon:
        pass


class BufferedLineDifference(BlockExtractionMethod):
    """ buffers each line string by a given epsilon, and returns the difference
    between the area and buffered road linestrings

    taken from: https://gis.stackexchange.com/a/58674

    suggested epsilons:
        1e-4 for generating representative graphics
        5e-6 for generating workable shapefiles
    """

    def __init__(self, epsilon: float = 5e-6, name: Optional[str] = None):
        self.epsilon = epsilon
        self.name    = name

    def __repr__(self):
        return "BufferedLineDifference(epsilon={})".format(self.epsilon)

    def extract(self, region: Union[Polygon, MultiPolygon], linestrings: MultiLineString) -> MultiPolygon:
        info("%sApplying buffer of %s.", "(" + self.name + ") " if self.name else "", self.epsilon)
        buffered = linestrings.buffer(self.epsilon)
        info("%sCalculating difference.", "(" + self.name + ") " if self.name else "")
        difference = region.difference(buffered)
        return MultiPolygon([difference]) if difference.type == "Polygon" else difference


class IntersectionPolygonization(BlockExtractionMethod):
    """ converts each linestring into multiple straight segments, and then uses GDAL's
    built-in polygonization function to return polygons

    taken from: https://peteris.rocks/blog/openstreetmap-city-blocks-as-geojson-polygons/#mapzen-metro-extracts://gis.stackexchange.com/a/58674

    poor results with non-quadrilateral blocks, generally works best for developed countries
    """

    def __repr__(self):
        return "IntersectionPolygonization"

    @staticmethod
    def get_line_feature(start, stop, properties):
        return {
            "type": "Feature",
            "properties": properties,
            "geometry": {
                "type": "LineString",
                "coordinates": [start, stop]
            }
        }

    @staticmethod
    def segment_streets(multipoint_lines):
        output = {
            "type": "FeatureCollection",
            "features": []
        }

        for feature in multipoint_lines['features']:
            output['features'] += [
                IntersectionPolygonization.get_line_feature(current, feature['geometry']['coordinates'][i+1], feature['properties'])
                for (i, current) in enumerate(feature['geometry']['coordinates'][:-1])]
        return output

    @staticmethod
    def polygonize_streets(streets):
        lines = []
        for feature in streets['features']:
            lines.append(asShape(feature['geometry']))

        polys = list(polygonize(lines))

        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        for poly in polys:
            geojson['features'].append({
                "type": "Feature",
                "properties": {},
                "geometry": mapping(poly)
            })

        return geojson

    def extract(self, region: Union[Polygon, MultiPolygon], linestrings: MultiLineString) -> MultiPolygon:
        # perform segmentation
        segmented_streets = IntersectionPolygonization.segment_streets(linestrings)
        # add the region boundary as an additional constraint
        constrained_linestrings = segmented_streets + [region.exterior]
        return MultiPolygon(list(IntersectionPolygonization.polygonize_streets(constrained_linestrings)))


DEFAULT_EXTRACTION_METHOD = BufferedLineDifference
