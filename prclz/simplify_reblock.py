from shapely.geometry import (LineString, LinearRing, 
                              MultiPolygon, Polygon, MultiLineString, 
                              Point, MultiPoint, LineString)
from shapely.ops import unary_union, linemerge
from typing import Callable, List 
import pandas as pd 
import numpy as np 
from itertools import combinations, chain, permutations
import tqdm 

import i_topology

'''
To-Do:
    - running final tests of simplify_reblocked_graph
'''

def to_gdf(geom):
    if isinstance(geom, list):
        return  gpd.GeoDataFrame.from_dict({'geometry': geom})
    else:
        return  gpd.GeoDataFrame.from_dict({'geometry': [geom]})
#####################################################################
## Functions for adding thru-streets after reblocking ###############
def edge_seq_to_linestring(e_seq, graph):

    lines = []
    for e in e_seq:
        e_line = LineString(graph.edge_to_coords(e))
        lines.append(e_line)
    return unary_union(lines)

def get_through_lines(
    planar_graph: i_topology.PlanarGraph,
    orig_metric_closure: i_topology.PlanarGraph,
    ratio_cutoff: float,
    cost_fn: Callable,
    ) -> List[LineString]:
    '''
    This adds through streets to a reblocked graph. The orig_metric_closure
    already contains the shortest paths (both path and distance) from
    the original graph. This fn extracts the optimal subgraph from the
    original graph, then calculates the new metric closure. If the ratio
    of the original shortest path distance to the new shortest path 
    distance is low enough, then we can connect these two vertices
    and increase connectivity, creating a through-street.

    NOTE: adds 'is_through_line' to PlanarGraph.es attributes
    '''
    steiner_edges = planar_graph.es.select(steiner_eq=True)
    opt_subgraph = planar_graph.subgraph_edges(steiner_edges)
    opt_metric_closure = i_topology.build_weighted_complete_graph(opt_subgraph,
                                                                  opt_subgraph.vs.select(terminal_eq=True),
                                                                  cost_fn)

    # Get ratio of orig/new shortest path distance
    combs_list = list(combinations(orig_metric_closure.vs, 2))
    for v0, v1 in combs_list:
        e_orig = orig_metric_closure.es.select(_within=[v0.index, v1.index])[0]
        e_opt = opt_metric_closure.es.select(_within=[v0.index, v1.index])[0]

        e_orig['ratio'] = e_orig['weight'] / e_opt['weight']

    # Get edges over a certain threshold, and add the 
    #     paths from the original metric closure to our reblock data
    cutoff = ratio_cutoff
    post_process_lines = []
    planar_graph.es['is_through_line'] = False
    for e in orig_metric_closure.es.select(ratio_lt = cutoff):
        edge_path = planar_graph.es[e['path']]
        edge_path['is_through_line'] = True 
        path_linestring = edge_seq_to_linestring(edge_path, planar_graph)
        post_process_lines.append(path_linestring)
    return post_process_lines
#####################################################################
#####################################################################


#####################################################################
## Functions to simplify linestrings after reblocking ###############
def simplify_linestring(init_ls: LineString,
                        admiss_region: Polygon,
                        ) -> LineString:
    '''
    Finds the simplest linestring maintaining the endpoints
    of the initial linestring, but only within the 
    admissable region
    '''

    EPS = 1e-6

    ambient_space = admiss_region.envelope.buffer(0.00001)
    invalid_region = ambient_space.difference(admiss_region.buffer(EPS))
    init_pts = pd.Series([Point(c) for c in init_ls.coords])

    if init_ls.intersects(invalid_region):
        print("Invalid region should not intersect orig line")
        return init_ls

    cur_pt_idxs = [0, len(init_pts)-1]
    cur_linestring = LineString(list(init_pts[cur_pt_idxs]))

    while cur_linestring.intersects(invalid_region):

        # Get the pt that is the farthest from the cur_linestring
        distances = [cur_linestring.distance(p) for p in init_pts]
        farthest_idx = np.argmax(distances)

        # Add it to our pt idx list and our linestring
        # NOTE: we maintain the idx list so the linestring
        #       is in the right order
        cur_pt_idxs.append(farthest_idx)
        cur_pt_idxs.sort()
        cur_linestring = LineString(list(init_pts[cur_pt_idxs]))

    return cur_linestring

def simplify_reblocked_graph(planar_graph: i_topology.PlanarGraph):

    # Extract the optimal subgraph of new lines only
    def filter(edge) -> bool:
        if 'is_through_line' in edge.attributes():
            return ((edge['steiner'] or edge['is_through_line']) and (edge['edge_type'] != 'highway'))
        else:
            return (edge['steiner'] and (edge['edge_type'] != 'highway'))

    opt_subgraph = planar_graph.subgraph_edges(planar_graph.es.select(filter))
    idx_v_pieces = opt_subgraph.to_pieces()
    simplified_linestrings = []
    #all_regions = []
    #all_lines = []
    print("Simplifing linestrings...")
    for v_indices in tqdm.tqdm(idx_v_pieces, total=len(idx_v_pieces)):
        # Convert vertices -> edges -> linestrings
        v_seq = opt_subgraph.vs[v_indices]
        edges = opt_subgraph.es.select(_within=v_seq)
        #edge_linestring = linemerge(edge_seq_to_linestring(edges, opt_subgraph))

        # And fetch the buffer area based on the width across 
        # that linestring/edge-sequence  

        edge_linestring, _, admiss_region, _ = opt_subgraph.get_steiner_linestrings(expand=False, return_polys=True, edge_seq=edges)
        edge_linestring = linemerge(edge_linestring)
        #all_lines.append(edge_linestring)
        #all_regions.append(admiss_region)
        #print("Type = {}".format(type(edge_linestring)))
        # if not isinstance(edge_linestring, LineString):
        #     return edge_linestring
        simplified_linestring = simplify_linestring(edge_linestring, admiss_region)
        simplified_linestrings.append(simplified_linestring)
    return simplified_linestrings
    #return all_lines, all_regions
#####################################################################
#####################################################################


# c0 = [(a,np.random.randn(1).item()) for a in range(5)]
# c1 = [(5,b) for b in range(6)]
# init_ls = LineString(c0+c1)
# admiss_region = init_ls.buffer(1)

# simpl = simplify_linestring(init_ls, admiss_region)