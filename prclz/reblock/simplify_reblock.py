from itertools import chain, combinations, permutations
from typing import Callable, List

import numpy as np
import pandas as pd
import tqdm
from shapely.geometry import (LinearRing, LineString, MultiLineString,
                              MultiPoint, MultiPolygon, Point, Polygon)
from shapely.ops import linemerge, unary_union

import i_topology

'''
To-Do:
    - running final tests of simplify_reblocked_graph
'''

# def to_gdf(geom):
#     if isinstance(geom, list):
#         return  gpd.GeoDataFrame.from_dict({'geometry': geom})
#     else:
#         return  gpd.GeoDataFrame.from_dict({'geometry': [geom]})

#####################################################################
# ## Functions to simplify linestrings after reblocking ###############
# def simplify_linestring(init_ls: LineString,
#                         admiss_region: Polygon,
#                         ) -> LineString:
#     '''
#     Finds the simplest linestring maintaining the endpoints
#     of the initial linestring, but only within the 
#     admissable region
#     '''

#     EPS = 1e-6

#     ambient_space = admiss_region.envelope.buffer(0.00001)
#     invalid_region = ambient_space.difference(admiss_region.buffer(EPS))
#     init_pts = pd.Series([Point(c) for c in init_ls.coords])

#     if init_ls.intersects(invalid_region):
#         print("Invalid region should not intersect orig line")
#         return init_ls

#     cur_pt_idxs = [0, len(init_pts)-1]
#     cur_linestring = LineString(list(init_pts[cur_pt_idxs]))

#     while cur_linestring.intersects(invalid_region):

#         # Get the pt that is the farthest from the cur_linestring
#         distances = [cur_linestring.distance(p) for p in init_pts]
#         farthest_idx = np.argmax(distances)

#         # Add it to our pt idx list and our linestring
#         # NOTE: we maintain the idx list so the linestring
#         #       is in the right order
#         cur_pt_idxs.append(farthest_idx)
#         cur_pt_idxs.sort()
#         cur_linestring = LineString(list(init_pts[cur_pt_idxs]))

#     return cur_linestring

# def simplify_reblocked_graph(planar_graph: i_topology.PlanarGraph):

#     # Extract the optimal subgraph of new lines only
#     def filter(edge) -> bool:
#         if 'is_through_line' in edge.attributes():
#             return ((edge['steiner'] or edge['is_through_line']) and (edge['edge_type'] != 'highway'))
#         else:
#             return (edge['steiner'] and (edge['edge_type'] != 'highway'))

#     opt_subgraph = planar_graph.subgraph_edges(planar_graph.es.select(filter))
#     idx_v_pieces = opt_subgraph.to_pieces()
#     simplified_linestrings = []

#     print("Simplifing linestrings...")
#     for v_indices in tqdm.tqdm(idx_v_pieces, total=len(idx_v_pieces)):
#         # Convert vertices -> edges -> linestrings
#         v_seq = opt_subgraph.vs[v_indices]
#         edges = opt_subgraph.es.select(_within=v_seq)

#         # And fetch the buffer area based on the width across 
#         # that linestring/edge-sequence  

#         edge_linestring, _, admiss_region, _ = opt_subgraph.get_steiner_linestrings(expand=False, return_polys=True, edge_seq=edges)
#         edge_linestring = linemerge(edge_linestring)
#         #all_lines.append(edge_linestring)
#         #all_regions.append(admiss_region)
#         #print("Type = {}".format(type(edge_linestring)))
#         # if not isinstance(edge_linestring, LineString):
#         #     return edge_linestring
#         simplified_linestring = simplify_linestring(edge_linestring, admiss_region)
#         simplified_linestrings.append(simplified_linestring)
#     return simplified_linestrings
#     #return all_lines, all_regions
#####################################################################
#####################################################################

