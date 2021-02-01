import typing 
from typing import Union, Sequence
import numpy as np 
import igraph 
from itertools import combinations, chain, permutations
from functools import reduce 
import pickle 
import os 
from shapely.geometry import LineString, LinearRing, MultiPolygon, Polygon, MultiLineString, Point, MultiPoint, LineString
from shapely.ops import cascaded_union, unary_union
from shapely.wkt import loads
import time 
import matplotlib.pyplot as plt 
import sys 
import argparse
import geopy.distance as gpy 
import pandas as pd 
import geopandas as gpd
from typing import List, Tuple
from heapdict import heapdict
#from fib_heap import Fibonacci_heap
import tqdm 
#from path_cost import BasePathCost

'''
TO-DO:
- Try different loss functions on DJI
- Build a visualizer
    - maybe one to include the width of the road,
      where the road is a polygon rather than linestring...
'''

from ..data_processing.setup_paths import build_data_dir, TRANS_TABLE

BUF_EPS = 1e-4
BUF_RATE = 2

def map_pts(pt_list, pt0, pt1):
    '''
    Helper function, assigns each pt in pt_list to whichever
    node it's closer to btwn u and v
    '''
    pt0_assign = []
    pt1_assign = []
    for pt in pt_list:
        if distance(pt, pt0) < distance(pt, pt1):
            pt0_assign.append(pt)
        else:
            pt1_assign.append(pt)
    assert len(pt0_assign) == len(pt1_assign), "Lens are not the same -- should be equal"
    return pt0_assign, pt1_assign


def shortest_path(g, 
                  origin_pt: Tuple[float, float], 
                  target_pt: Tuple[float, float], 
                  cost_fn,
                  return_epath = True,
                  return_dist = True):

    #print("TEST fdafdafdsa INSIDE")
    
    # Build priority queue
    Q = heapdict()
    #Q = Fibonacci_heap()
    for v in g.vs:
        if v['name'] == origin_pt:
            d = 0
            path = None
        else:
            d = np.inf
            path = None
        v['_dji_dist'] = d 
        v['_dji_prev_idx'] = path 
        v['_visited'] = False
        if v['name'] == origin_pt:
            Q[v] = v['_dji_dist']
            #v['_ref'] = Q.enqueue(v, v['_dji_dist'])
        

    # Initialize current node
    cur_v, _ = Q.popitem()
    #cur_v = Q.dequeue_min().get_value()
    cur_pt = cur_v['name']
    start_idx = cur_v.index
    while cur_pt != target_pt:
        #print("cur_pt = {}".format(cur_pt))
        for next_v in cur_v.neighbors():
            #print("\tneighbor = {}".format(n['name']))
            if next_v['_visited']:
                #print("\t\talready visited -- continue")
                continue
            else:
                if cur_v['_dji_prev_idx'] is None:
                    prev_v = None
                else:
                    prev_v = g.vs[cur_v['_dji_prev_idx']]
                marg_dist = cost_fn(g, cur_v, next_v, prev_v)
                #marg_dist = distance(cur_pt, next_v['name'])
                new_dist = marg_dist + cur_v['_dji_dist']
                if new_dist < next_v['_dji_dist']:
                    # print("\t\tresetting path {} to {}".format(cur_v['_dji_prev_idx'], cur_v.index))
                    # print("\t\tdist from {} to {}".format(n['_dji_dist'], new_dist))
                    next_v['_dji_dist'] = new_dist
                    next_v['_dji_prev_idx'] = cur_v.index

                    # Update the priority in Q
                    Q[next_v] = new_dist 
                    
        cur_v['_visited'] = True
        cur_v, _ = Q.popitem()
        #cur_v = Q.dequeue_min().get_value()
        cur_pt = cur_v['name']
        cur_idx = cur_v.index 

    #print("Found {} at {}: go back to idx {}".format(target_pt, cur_pt, cur_v['_dji_path']))

    # Clean up returned output
    rv = []
    if return_epath:
        vpath_indices = []
        epath_indices = []
        while cur_idx != start_idx:
            vpath_indices.append(cur_idx)
            parent_idx = cur_v['_dji_prev_idx']
            #print("Goal = {} | Between v{}-v{}".format(start_idx, cur_idx, parent_idx))
            cur_e = g.es.find(_between=((cur_idx,), (parent_idx,))).index
            #print("Between v{}-v{} is edge {}\n".format(cur_idx, parent_idx, cur_e))
            epath_indices.append(cur_e)
            cur_v = g.vs[parent_idx]
            cur_idx = cur_v.index
        rv.append(epath_indices)

    if return_dist:
        vpath_dist = g.vs.select(name=target_pt)['_dji_dist'][0]
        rv.append(vpath_dist)
    return rv 


def build_weighted_complete_graph(G: igraph.Graph, 
                                  terminal_vertices: igraph.EdgeSeq,
                                  cost_fn):
    H = PlanarGraph()
    combs_list = list(combinations(terminal_vertices, 2))
    print("Building metric closure...")
    for u,v in tqdm.tqdm(combs_list, total=len(combs_list)):
    #for u,v in combs_list:
        if not isinstance(u, tuple):
            u = u['name']
            v = v['name']
        #print("in build_weighted_complete_graph: {},{}".format(u, v))
        path_idxs, path_distance = shortest_path(g=G, origin_pt=u, 
                                                 target_pt=v, cost_fn=cost_fn)
        #print("...done")
        path_edges = G.es[path_idxs]
        kwargs = {'weight':path_distance, 'path':path_idxs}
        H.add_edge(u, v, **kwargs)
    return H 

def flex_steiner_tree(G: igraph.Graph, 
                      terminal_vertices: igraph.EdgeSeq,
                      cost_fn,
                      return_metric_closure: bool = False):
    '''
    terminal_nodes is List of igraph.Vertex
    '''

    # (1) Build closed graph of terminal_vertices where each weight is the shortest path distance
    H = build_weighted_complete_graph(G, 
                                      terminal_vertices, 
                                      cost_fn=cost_fn)

    # (2) Now get the MST of that complete graph of only terminal_vertices
    if "weight" not in H.es.attributes():
        print("----H graph does not have weight, ERROR")
        print("\t\t there are {}".format(len(terminal_vertices)))
    mst_edge_idxs = H.spanning_tree(weights='weight', return_tree=False)

    # Now, we join the paths for all the mst_edge_idxs
    steiner_edge_idxs = list(chain.from_iterable(H.es[i]['path'] for i in mst_edge_idxs))

    if return_metric_closure:
        return steiner_edge_idxs, H
    else:
        return steiner_edge_idxs

def shortest_path_orig(g, 
                  origin_pt, 
                  target_pt):
    
    # Create bool attr for visited
    g.vs['visited'] = False
    #visited_indices = set()

    # Create dist attr
    g.vs['_dji_dist'] = np.inf 
    g.vs['_dji_path'] = [[]*len(g.vs['_dji_dist'])]
    g.vs.select(name=origin_pt)['_dji_dist'] = 0
    g.vs.select(name=origin_pt)['_dji_path'] = [[origin_pt]]

    # Initialize current node
    cur_pt = origin_pt
    cur_v = g.vs.select(name=cur_pt)[0]
    while cur_pt != target_pt:
        #print("ON {}".format(cur_pt))
        for n in cur_v.neighbors():
            if n['visited']:
                continue
            else:
                new_dist = cur_v['_dji_dist'] + distance(cur_pt, n['name'])
                if new_dist < n['_dji_dist']:
                    #print("\tupdating dist to {} = {}".format(n['name'], new_dist))
                    n['_dji_dist'] = new_dist

                    # print(cur_v['_dji_path'])
                    n['_dji_path'] = cur_v['_dji_path'][:]
                    n['_dji_path'].append(n['name'])
                    
        cur_v['visited'] = True
        argmin = np.argmin([x['_dji_dist'] for x in g.vs.select(visited=False)])
        cur_v = list(g.vs.select(visited=False))[argmin]
        cur_pt = cur_v['name']

###############################################################################
###############################################################################

def angle_btwn(pt0, pt1, pt2, degrees=False) -> float:
    pt0 = np.array(pt0)
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)

    pt0 -= pt1
    pt2 -= pt1 

    unit_vec0 = pt0 / np.linalg.norm(pt0)
    unit_vec2 = pt2 / np.linalg.norm(pt2)
    dot = np.dot(unit_vec0, unit_vec2)
    dot = np.clip(dot, -1, 1)
    angle = np.arccos(dot)
    if np.isnan(angle):
        print("dot = {}".format(dot))
    if degrees:
        angle = np.degrees(angle)
    return angle

def igraph_steiner_tree(G, terminal_vertices, weight='weight'):
    '''
    terminal_nodes is List of igraph.Vertex
    '''

    # Build closed graph of terminal_vertices where each weight is the shortest path distance
    H = PlanarGraph()
    for u,v in combinations(terminal_vertices, 2):
        path_idxs = G.get_shortest_paths(u, v, weights='weight', output='epath')
        path_edges = G.es[path_idxs[0]]
        path_distance = reduce(lambda x,y : x+y, map(lambda x: x['weight'], path_edges))
        kwargs = {'weight':path_distance, 'path':path_idxs[0]}
        H.add_edge(u['name'], v['name'], **kwargs)

    # Now get the MST of that complete graph of only terminal_vertices
    if "weight" not in H.es.attributes():
        print("----H graph does not have weight, ERROR")
        print("\t\t there are {}".format(len(terminal_vertices)))
    mst_edge_idxs = H.spanning_tree(weights='weight', return_tree=False)

    # Now, we join the paths for all the mst_edge_idxs
    steiner_edge_idxs = list(chain.from_iterable(H.es[i]['path'] for i in mst_edge_idxs))

    return steiner_edge_idxs

def distance_meters(a0, a1):

    lonlat_a0 = gpy.lonlat(*a0)
    lonlat_a1 = gpy.lonlat(*a1)

    return gpy.distance(lonlat_a0, lonlat_a1).meters

def distance(a0, a1):

    if not isinstance(a0, np.ndarray):
        a0 = np.array(a0)
    if not isinstance(a1, np.ndarray):
        a1 = np.array(a1)

    return np.sqrt(np.sum((a0-a1)**2))

def min_distance_from_point_to_line(coords, edge_tuple):
    '''
    Just returns the min distance from the edge to the node

    Inputs:
        - coords (tuple) coordinate pair
        - edge_tuple (tuple of tuples) or coordinate end points defining a line
    '''
    x1,y1 = edge_tuple[0]
    x2,y2 = edge_tuple[1]
    x0,y0 = coords 

    num = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    den = np.sqrt((y2-y1)**2 + (x2-x1)**2)

    return num/den     

def node_on_edge(edge_tuple, coords):
    '''
    Because line segments are finite, when calculating min distance from edge
    to a point we need to check whether the projection onto the LINE defined by
    the edge is in fact on the edge or outside of it
    
    Inputs:
        - coords (tuple) coordinate pair
        - edge_tuple (tuple of tuples) or coordinate end points defining a line
    '''

    mid_x = (edge_tuple[0][0]+edge_tuple[1][0]) / 2.
    mid_y = (edge_tuple[0][1]+edge_tuple[1][1]) / 2.
    mid_coords = (mid_x, mid_y)

    # NOTE: the distance from the midpoint of the edge to any point on the edge
    #       cannot be greater than the dist to the end points
    np_coords = np.array(coords)
    np_mid_coords = np.array(mid_coords)

    max_distance = distance(np.array(edge_tuple[0]), np_mid_coords)
    qc0 = distance(np_mid_coords, np.array(edge_tuple[0]))
    qc1 = distance(np_mid_coords, np.array(edge_tuple[1]))
    assert np.sum(np.abs(qc0-qc1)) < 10e-4, "NOT TRUE MIDPOINT"

    node_distance = distance(np_coords, np_mid_coords)

    if node_distance > max_distance:
        return False
    else:
        return True 

def vector_projection(edge_tuple, coords):
    '''
    Returns the vector projection of node onto the LINE defined
    by the edge
    https://en.wikipedia.org/wiki/Vector_projection
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    '''
    a_vector = np.array(coords)

    b_vector = np.array(edge_tuple[0]) - np.array(edge_tuple[1])

    b_unit = b_vector / np.linalg.norm(b_vector)

    b_normal = np.array([-b_unit[1], b_unit[0]])

    if not(np.abs(np.sum(b_normal*b_unit)) < 10e-4):
        print()
        print("a_vector = ", a_vector)
        print("b_vector = ", b_vector)
        print("b_normal = ", b_normal)
        print("b_unit = ", b_unit)
        print()

    assert np.abs(np.sum(b_normal*b_unit)) < 10e-4, "b_normal and b_unit are not orthog"

    #min_distance = self.min_distance_to_node(node)
    min_distance = min_distance_from_point_to_line(coords, edge_tuple)

    # Depending on the ordering the +/- can get reversed so this is 
    # just a little hacky workaround to make it 100% robust
    proj1 = a_vector + min_distance * b_normal
    proj2 = a_vector - min_distance * b_normal

    if min_distance_from_point_to_line(proj1, edge_tuple) < 10e-4:
        return (proj1[0], proj1[1])
    elif min_distance_from_point_to_line(proj2, edge_tuple) < 10e-4:
        return (proj2[0], proj2[1])
    else:
        assert False, "Vector projection failed"


class PlanarGraph(igraph.Graph):

    @staticmethod
    def from_edges(edges):
        graph = PlanarGraph()
        for edge in edges:
            graph.add_edge(*edge)
        return graph

    @staticmethod
    def linestring_to_planar_graph(linestring: Union[LineString, Polygon], append_connection=True):
        '''
        Helper function to convert a single Shapely linestring
        to a PlanarGraph
        '''

        # linestring -> List[Nodes]
        if isinstance(linestring, LineString):
            nodes = linestring.coords
        elif isinstance(linestring, Polygon):
            nodes = linestring.exterior.coords
        else:
            assert False, "Hacky error - invalid type!"

        # List[Nodes] -> List[Edges]
        if append_connection:
            nodes.append(nodes[0])
        edges = []
        for i, n in enumerate(nodes):
            if i==0:
                continue
            else:
                edges.append( (n, nodes[i-1]) )

        # List[Edges] -> PlanarGraph
        pgraph = PlanarGraph.from_edges(edges)

        return pgraph 

    @staticmethod
    def multipolygon_to_planar_graph(multipolygon: MultiPolygon):
        '''
        Helper function to convert a Shapely multipolygon
        to a PlanarGraph
        '''
        print("Creating PlanarGraph from multipolygon")

        pgraph = PlanarGraph()

        for parcel_id, polygon in enumerate(multipolygon):
            nodes = polygon.exterior.coords 
            for i, n in enumerate(nodes):
                if i != 0:
                    pgraph.add_edge(n, prev_n, parcel_id=parcel_id)
                prev_n = n 
        return pgraph 


    @staticmethod
    def multilinestring_to_planar_graph(multilinestring: MultiLineString):
        '''
        Helper function to convert a Shapely multilinestring
        to a PlanarGraph
        '''

        pgraph = PlanarGraph()

        for linestring in multilinestring:
            # linestring -> List[Nodes]
            #nodes = [Node(p) for p in linestring.coords]
            nodes = list(linestring.coords)

            # List[Nodes] -> List[Edges]
            #nodes.append(nodes[0])
            for i, n in enumerate(nodes):
                if i==0:
                    continue
                else:
                    pgraph.add_edge(n, nodes[i-1])

        return pgraph

    @staticmethod
    def load_planar(file_path):
        '''
        Loads a planar graph from a saved via
        '''

        # The mapping to recover coord is stored separately
        file_path_mapping = file_path+".dict"
        assert os.path.isfile(file_path_mapping), "There should be a corresponding .dict file associated with the graphml file"
        with open(file_path_mapping, 'rb') as file:
            idx_mapping = pickle.load(file)

        # Now load the graphML file and join
        graph = PlanarGraph.Read_GraphML(file_path)
        graph.vs['name'] = [idx_mapping[i] for i in graph.vs['id']]
        del graph.vs['id']

        return graph

    def save_planar(self, file_path):
        '''
        Pickling the object wasn't working and saving as
        GraphML does. However, this can only maintain simple
        boolean, string, numeric attributes so we create a dictionary
        which can recover the lost coordinate pairs (which are python tuples) 
        '''

        # with open(file_path, 'wb') as file:
        #     pickle.dump(self, file)
        # Save out idx->coord mapping
        idx_mapping = {}
        for i, v in enumerate(self.vs):
            idx = "n{}".format(i)
            idx_mapping[idx] = v['name']
        file_path_mapping = file_path+".dict"
        with open(file_path_mapping, 'wb') as file:
            pickle.dump(idx_mapping, file)

        # Save out the graph
        self.save(file_path, format='graphml')


    def add_node(self, coords, terminal=False):
        '''
        Adds coords to the graph but checks if coords are already in
        graph
        '''

        if len(self.vs) == 0:
            self.add_vertex(name=coords, terminal=terminal)
        else:
            seq = self.vs.select(name=coords)
            if len(seq) == 0:
                self.add_vertex(name=coords, terminal=terminal)
            elif len(seq) == 1:
                seq[0]['terminal'] = terminal 
            elif len(seq) > 1:
                assert False, "Hacky error - there are duplicate nodes in graph"

    def add_edge(self, coords0, coords1, 
                 terminal0=False, terminal1=False, 
                 parcel_id=None, **kwargs):
        '''
        Adds edge to the graph but checks if edge already exists. Also, if either
        coords is not already in the graph, it adds them
        '''

        # Safely add nodes
        self.add_node(coords0, terminal0)
        self.add_node(coords1, terminal1)

        v0 = self.vs.select(name=coords0)
        v1 = self.vs.select(name=coords1)

        # Safely add edge after checking whether edge exists already
        edge_seq = self.es.select(_between=(v0, v1))
        if len(edge_seq) == 0:
            kwargs['steiner'] = False
            if "weight" not in kwargs.keys():
                kwargs['weight'] = distance(coords0, coords1)
                kwargs['eucl_dist'] = distance(coords0, coords1)
            super().add_edge(v0[0], v1[0], **kwargs)

        if parcel_id is not None:
            edge_seq = self.es.select(_between=(v0,v1))
            if 'parcel_id' not in edge_seq.attributes():
                self.es['parcel_id'] = None 
            if edge_seq['parcel_id'][0] is None:
                edge_seq['parcel_id'] = [{parcel_id}]
            else:
                edge_seq['parcel_id'][0].add(parcel_id)    


    def split_edge_by_node(self, edge_tuple, coords, terminal=False):
        '''
        Given an existing edge btwn 2 nodes, and a third unconnected node, 
        replaces the existing edge with 2 new edges with the previously
        unconnected node between the two
        NOTE: if the new node is already one of the edges, we do not create a self-edge

        Inputs:
            - edge_tuple: two coord pairs ex. [(0,1), (1,1)]
            - coords: coord pair ex. (2,2)
        '''
        orig_coords0, orig_coords1 = edge_tuple
        if coords == orig_coords0:
            self.vs.select(name=orig_coords0)['terminal'] = terminal
        elif coords == orig_coords1:
            self.vs.select(name=orig_coords1)['terminal'] = terminal
        else:
            orig_vtx0 = self.vs.select(name=orig_coords0)
            orig_vtx1 = self.vs.select(name=orig_coords1)
            assert len(orig_vtx0) == 1, "Found {} vertices in orig_vtx0".format(len(orig_vtx0))
            assert len(orig_vtx1) == 1, "Found {} vertices in orig_vtx1".format(len(orig_vtx1))
            edge_seq = self.es.select(_between=(orig_vtx0, orig_vtx1))
            super().delete_edges(edge_seq)

            self.add_edge(orig_coords0, coords, terminal1=terminal)
            self.add_edge(coords, orig_coords1, terminal0=terminal)

    @staticmethod
    def closest_point_to_node(edge_tuple, coords):
        '''
        The edge_tuple specifies an edge and this returns the point on that
        line segment closest to 
        '''

        projected_node = vector_projection(edge_tuple, coords)
        if node_on_edge(edge_tuple, projected_node):
            return projected_node
        else:
            dist_node0 = distance(edge_tuple[0], coords)
            dist_node1 = distance(edge_tuple[1], coords)
            if dist_node0 <= dist_node1:
                return edge_tuple[0]
            else:
                return edge_tuple[1]

    def coords_to_edge(self, 
                       coord0: Tuple[float, float], 
                       coord1: Tuple[float, float]) -> igraph.EdgeSeq:
        '''
        If we have the coordinates of two nodes, returns
        the edge 
        '''
        v0 = self.vs.select(name=coord0)
        v1 = self.vs.select(name=coord1)
        edge = self.es.select(_between=(v0, v1))
        return edge 

    def coord_path_to_edge_path(self, 
                                coord_seq: List[Tuple[float, float]]) -> List[igraph.Edge]:
        '''
        Converts a path specified by coordinates to an edge sequence
        '''
        edge_list = []
        for c0, c1 in zip(coord_seq, coord_seq[1:]):
            edge = self.coords_to_edge(c0, c1)[0]
            edge_list.append(edge)
        return edge_list 

    def edges_in_parcel(self, parcel_id: int) -> igraph.EdgeSeq:
        '''
        Given a parcel_id, returns edge seq of
        all edges associated with that parcel. 
        NOTE: an edge is either in 1-2 parcels
        '''
        def fn(e: igraph.Edge):
            if e['parcel_id'] is None:
                return False
            else:
                return (parcel_id in e['parcel_id'])
        return self.es.select(fn)

    def edge_to_coords(self, edge, expand=False):
        '''
        Given an edge, returns the edge_tuple of
        the corresponding coordinates

        NOTE: if we have simplified the graph then we need
              to unpack the nodes which are saved within
              the 'path' attribute 
        '''

        v0_idx, v1_idx = edge.tuple 
        v0_coords = self.vs[v0_idx]['name']
        v1_coords = self.vs[v1_idx]['name']
        if expand:
            edge_tuple = [v0_coords] + edge['path'] + [v1_coords]
        else:
            edge_tuple = (v0_coords, v1_coords)

        return edge_tuple 

    def setup_linestring_attr(self):
        if 'linestring' not in self.es.attributes():
            self.es['linestring'] = [LineString(self.edge_to_coords(e)) for e in self.es]
        else:
            no_linestring_attr = self.es.select(linestring_eq=None)
            no_linestring_attr['linestring'] = [LineString(self.edge_to_coords(e)) for e in no_linestring_attr]

    def cleanup_linestring_attr(self):
        del self.es['linestring']

    def find_candidate_edges(self, coords):

        self.setup_linestring_attr()

        point = Point(*coords)

        # Initialize while loop
        buf = BUF_EPS
        buffered_point = point.buffer(buf)
        edges = self.es.select(lambda e: e['linestring'].intersects(buffered_point))
        i = 0
        while len(edges) == 0:
            buf *= BUF_RATE
            buffered_point = point.buffer(buf)
            edges = self.es.select(lambda e: e['linestring'].intersects(buffered_point))
            i += 1
        #print("Found {}/{} possible edges thru {} tries".format(len(edges), len(self.es), i))
        return edges 

    def add_bldg_centroid(self, pt: Point, e: igraph.Edge):
        '''
        Adding the building centroid at pt to the edge, e. First,
        find the point on e closest to the pt, and that's where
        we add.
        '''

        edge_coord_tuple = [v['name'] for v in e.vertex_tuple]
        pt_coord_tuple = list(pt.coords)[0]
        #print("edge_coord_tuple: {}".format(edge_coord_tuple))
        #print("edge_coord_tuple type: {}".format(type(edge_coord_tuple)))
        closest_node = PlanarGraph.closest_point_to_node(edge_coord_tuple, pt_coord_tuple)

        # Now add it
        self.split_edge_by_node(edge_coord_tuple, closest_node, terminal=True)


    def add_node_to_closest_edge(self, coords, terminal=False, fast=True, get_edge=False):
        '''
        Given the input node, this finds the closest point on each edge to that input node.
        It then adds that closest node to the graph. It splits the argmin edge into two
        corresponding edges so the new node is fully connected
        '''
        closest_edge_nodes = []
        closest_edge_distances = []

        if fast:
            cand_edges = self.find_candidate_edges(coords)
        else:
            cand_edges = self.es 

        for edge in cand_edges:

            edge_tuple = self.edge_to_coords(edge)

            #Skip self-edges
            if edge.is_loop():
                #print("\nSKIPPING EDGE BC ITS A SELF-EDGE\n")
                continue 

            closest_node = PlanarGraph.closest_point_to_node(edge_tuple, coords)
            closest_distance = distance(closest_node, coords)

            closest_edge_nodes.append(closest_node)
            closest_edge_distances.append(closest_distance)

        argmin = np.argmin(closest_edge_distances)

        closest_node = closest_edge_nodes[argmin]
        closest_edge = self.edge_to_coords(cand_edges[argmin])
        if get_edge:
            dist_meters = distance_meters(coords, closest_node)
            return cand_edges[argmin], dist_meters

        # Now add it
        self.split_edge_by_node(closest_edge, closest_node, terminal=terminal)

    def steiner_tree_approx(self, verbose=False):
        '''
        All Nodes within the graph have an attribute, Node.terminal, which is a boolean
        denoting whether they should be included in the set of terminal_nodes which
        are connected by the Steiner Tree approximation
        '''
        terminal_nodes = self.vs.select(terminal_eq=True)

        steiner_edge_idxs = igraph_steiner_tree(self, terminal_nodes)
        for i in steiner_edge_idxs:
            self.es[i]['steiner'] = True 

    def flex_steiner_tree_approx(self, 
                                 cost_fn, 
                                 return_metric_closure: bool = True):
        '''
        All Nodes within the graph have an attribute, Node.terminal, which is a boolean
        denoting whether they should be included in the set of terminal_nodes which
        are connected by the Steiner Tree approximation
        '''
        terminal_nodes = self.vs.select(terminal_eq=True)

        rv = flex_steiner_tree(self, 
                               terminal_nodes, 
                               cost_fn = cost_fn,
                               return_metric_closure = return_metric_closure)

        if return_metric_closure:
            steiner_edge_idxs, metric_closure = rv
        else:
            steiner_edge_idxs = rv  
            metric_closure = None  

        for i in steiner_edge_idxs:
            self.es[i]['steiner'] = True 

        return metric_closure


    def plot_reblock(self, output_file, visual_style=None):

        if visual_style is None:
            visual_style = {}
        
        vtx_color_map = {True: 'red', False: 'blue'}
        edg_color_map = {True: 'red', False: 'blue'}
        
        if 'terminal' in self.vs.attributes():
            if 'vertex_color' not in visual_style.keys():
                visual_style['vertex_color'] = [vtx_color_map[t] for t in self.vs['terminal'] ]
        
        if 'steiner' in self.es.attributes():
            if 'edge_color' not in visual_style.keys():
                visual_style['edge_color'] = [edg_color_map[t] for t in self.es['steiner'] ]
                #print(visual_style)

        if 'layout' not in visual_style.keys():
            visual_style['layout'] = [(x[0],-x[1]) for x in self.vs['name']]
        else:
            print("Layout is already in visual_style")
            
        if 'vertex_label' not in visual_style.keys():
            visual_style['vertex_label'] = [str(x) for x in self.vs['name']]
        else:
            print("vertex_label is already in visual_style")

        #return visual_style

        #print("visual_style = {}".format(visual_style))
        igraph.plot(self, output_file, **visual_style)

    def get_steiner_linestrings(self, 
                                expand=True,
                                return_polys=False,
                                edge_seq: igraph.EdgeSeq = None) -> MultiLineString:
        '''
        Takes the Steiner optimal edges from g and converts them

        If return_polys==True, will retur
        '''
        existing_lines = []
        new_lines = []
        
        existing_polys = []
        new_polys = []
        
        self.vs['_tmp_pts'] = None 
        if edge_seq is None:
            edge_seq = self.es 
        for e in edge_seq:

            if 'is_through_line' in self.es.attributes():
                is_opt = e['steiner'] or e['is_through_line']
            else:
                is_opt = e['steiner']
            
            if is_opt:
                #if e['edge_type'] == 'highway':
                path = LineString(self.edge_to_coords(e, expand))
                if e['weight'] == 0:
                    existing_lines.append(path)
                else:
                    new_lines.append(path)

                if return_polys:
                    #poly = self.buffer_by_width(e, e_path=path, expand=expand)
                    w = e['width']
                    path_left = list(path.parallel_offset(w, 'left').coords)
                    path_right = list(path.parallel_offset(w, 'right').coords)
                    poly_pts = path_left + path_right
                    poly = Polygon(poly_pts)
                    
                    # Now add the pts to the vertex
                    v0, v1 = e.vertex_tuple
                    v0_assign, v1_assign = map_pts(poly_pts, v0['name'], v1['name'])
                    if v0['_tmp_pts'] is None:
                        v0['_tmp_pts'] = []
                    if v1['_tmp_pts'] is None:
                        v1['_tmp_pts'] = []
                    v0['_tmp_pts'].extend(v0_assign)
                    v1['_tmp_pts'].extend(v1_assign)

                    if e['weight'] == 0:
                        existing_polys.append(poly)
                    else:
                        new_polys.append(poly)
        
        if return_polys:
            for e in edge_seq:
                if 'is_through_line' in self.es.attributes():
                    is_opt = e['steiner'] or e['is_through_line']
                else:
                    is_opt = e['steiner']
                #if e['steiner']:
                if is_opt:
                    v0, v1 = e.vertex_tuple
                    for v in [v0, v1]:
                        if len(v['_tmp_pts']) == 4:
                            pts = v['_tmp_pts']
                            rect = Polygon(v['_tmp_pts']).convex_hull
                            
                            if e['weight'] == 0:
                                existing_polys.append(rect)
                            else:
                                new_polys.append(rect)    

        #lines = [LineString(self.edge_to_coords(e)) for e in self.es if e['steiner']]
        new_lines = unary_union(new_lines)
        existing_lines = unary_union(existing_lines)
        
        if return_polys:
            new_polys = unary_union(new_polys)
            existing_polys = unary_union(existing_polys)
        
            return new_lines, existing_lines, new_polys, existing_polys
        else:
            return new_lines, existing_lines

    def get_terminal_points(self) -> MultiPoint:
        '''
        Takes all the terminal nodes (ie buildings) and returns them as 
        shapely MultiPoint
        '''
        points = [Point(v['name']) for v in self.vs if v['terminal']]
        multi_point = unary_union(points)
        return multi_point

    def get_linestrings(self) -> MultiLineString:
        '''
        Takes the Steiner optimal edges from g and converts them
        '''
        lines = [LineString(self.edge_to_coords(e)) for e in self.es]
        multi_line = unary_union(lines)
        return multi_line 

    # These methods are for simplifying the graph
    def simplify_node(self, vertex):
        '''
        If we simplify node B with connections A -- B -- C
        then we end up with (AB) -- C where the weight 
        of the edge between (AB) and C equals the sum of the
        weights between A-B and B-C

        NOTE: this allows the graph to simplify long strings of nodes
        '''

        # Store the 2 neighbors of the node we are simplifying
        n0_vtx, n1_vtx = vertex.neighbors()
        n0_name = n0_vtx['name']
        n1_name = n1_vtx['name']
        n0_seq = self.vs.select(name=n0_vtx['name'])
        n1_seq = self.vs.select(name=n1_vtx['name'])
        v = self.vs.select(name=vertex['name'])

        # Grab each neighbor edge weight
        edge_n0 = self.es.select(_between=(n0_seq, v))
        edge_n1 = self.es.select(_between=(n1_seq, v))
        total_weight = edge_n0[0]['weight'] + edge_n1[0]['weight']

        # Form a new edge between the two neighbors
        # The new_path must reflect the node that will be removed and the
        #    2 edges that will be removed
        new_path = edge_n0[0]['path'] + [vertex['name']] + edge_n1[0]['path']
        super().add_edge(n0_seq[0], n1_seq[0], weight=total_weight, path=new_path)

        # Now we can delete the vertex and its 2 edges
        edge_n0 = self.es.select(_between=(n0_seq, v))
        super().delete_edges(edge_n0)

        edge_n1 = self.es.select(_between=(n1_seq, v))
        super().delete_edges(edge_n1)
        super().delete_vertices(v)

    def simplify(self):
        '''
        Many nodes exist to approximate curves in physical space. Calling this
        collapses those nodes to allow for faster downstream computation
        '''
        if 'path' not in self.vs.attributes():
            self.es['path'] = [ [] for v in self.vs]

        for v in self.vs:
            num_neighbors = len(v.neighbors())
            if num_neighbors == 2 and not v['terminal']:
                #print("simplifying node {}".format(v['name']))
                self.simplify_node(v)

    def search_continuous_edge(self, v, visited_indices=None):
        '''
        Given a vertex, v, will find the continuous string of
        vertices v is part of.
        '''

        if visited_indices is None:
            visited_indices = []
        
        neighbors = v.neighbors()
        visited_indices.append(v.index)

        if len(neighbors) != 2:
            return visited_indices
        else:
            for n in neighbors:
                if n.index not in visited_indices:
                    path = self.search_continuous_edge(n, visited_indices)
            return visited_indices 

    def to_pieces(self) -> List[List[int]]:
        '''
        Creates representation of graph 
        '''

        piece_list = []
        all_visited_idxs = set()

        print("Breaking to pieces...")
        for v in tqdm.tqdm(self.vs):
            if v.index not in all_visited_idxs:
                neighbors = v.neighbors()
                num_neighbors = len(neighbors)
                if num_neighbors == 2:
                    cont_vs = self.search_continuous_edge(v)
                    for v in cont_vs:
                        all_visited_idxs.add(v)
                    piece_list.append(cont_vs)
        return piece_list 



    def simplify_edge_width(self):
        '''
        Resets edge width to be the min over
        a continuous segment
        '''

        all_visited = set()

        print("Simplifying edge width...")
        for v in tqdm.tqdm(self.vs):
            neighbors = v.neighbors()
            num_neighbors = len(neighbors)
            if num_neighbors == 2:
                # get the two edges
                cont_vs = self.search_continuous_edge(v)
                edges = self.es.select(_within=cont_vs)
                min_width = min([e['width'] for e in edges])
                if min_width is None:
                    print("Vertex {} has {} neighbors".format(v.index, num_neighbors))
                    print("\tsegment is {}".format(cont_vs))
                for e in edges:
                    e['width'] = min_width

    ################################################### 
    ## ADDITIONAL ATTRIBUTES FOR ADVANCED REBLOCKING ##
    def set_edge_width(self, 
                       other_geoms: List[Polygon],
                       simplify=True) -> None:
        '''
        Adds following properties:
            To edges:
                - edge_width: min dist to other_geoms
        '''
        for e in self.es:
            e_line = LineString(self.edge_to_coords(e))
            distances = [e_line.distance(g) for g in other_geoms]
            e['width'] = min(distances) 
        if simplify:
            self.simplify_edge_width()  

    def set_node_angles(self, save_degree=True, format='degrees'):
        '''
        Adds following properties:
            To vertices:
                - angles: dict mapping nbors to angle
                - degree: number of neighbors
        '''
        assert format in {'degrees', 'radians'}, "Degree format must be one of |degrees| or |radians|"
        as_degrees = format == 'degrees'
        for v in self.vs:
            neighbors = v.neighbors()
            angle_dict = {}
            for n0 in neighbors:
                for n1 in neighbors:
                    if n0 is n1:
                        continue
                    elif (n1, n0) in angle_dict:
                        continue
                    else:
                        angle = angle_btwn(n0['name'], v['name'], n1['name'], degrees=as_degrees)
                        angle_dict[(n0.index, n1.index)] = angle 
            v['angles'] = angle_dict 
            if save_degree:
                v['degree'] = len(neighbors)
    ###################################################
        


def convert_to_lines(planar_graph) -> MultiLineString:
    lines = [LineString(planar_graph.edge_to_coords(e)) for e in planar_graph.es]
    multi_line = unary_union(lines)
    return multi_line 


def find_edge_from_coords(g, coord0, coord1):
    '''
    Given a pair of coordinates, checks whether the graph g
    contains an edge between that coordinate pair
    '''
    v0 = g.vs.select(name_eq=coord0)
    v1 = g.vs.select(name_eq=coord1)
    if len(v0)==0 or len(v1)==0:
        return None 
    else:
        edge = g.es.select(_between=(v0, v1))
        if len(edge)==0:
            return None 
        else:
            return edge[0]

########################################################################
# BELOW THIS, CODE IS BEING COMMENTED OUT TO LATER BE DELETED ##########
########################################################################


#def plot_edge_type(g, output_file):

#     edge_color_map = {None: 'red', 'waterway': 'blue', 
#                       'highway': 'black', 'natural': 'green', 'gadm_boundary': 'orange'}
#     visual_style = {}       
#     SMALL = 0       
#     visual_style['vertex_size'] = [SMALL for _ in g.vs]

#     if 'edge_type' not in g.es.attributes():
#         g.es['edge_type'] = None 
#     visual_style['edge_color'] = [edge_color_map[t] for t in g.es['edge_type'] ]
#     visual_style['layout'] = [(x[0],-x[1]) for x in g.vs['name']]

#     return igraph.plot(g, output_file, **visual_style)


# def plot_reblock(g, output_file):
#     vtx_color_map = {True: 'red', False: 'blue'}
#     edg_color_map = {True: 'red', False: 'blue'}
    
#     visual_style = {}
#     if 'vertex_color' not in visual_style.keys():
#         visual_style['vertex_color'] = [vtx_color_map[t] for t in g.vs['terminal'] ]
    
#     BIG = 20
#     SMALL = 20
#     if 'bbox' not in visual_style.keys():
#         visual_style['bbox'] = (900,900)
#     if 'vertex_size' not in visual_style.keys():
#         visual_style['vertex_size'] = [BIG if v['terminal'] else SMALL for v in g.vs]

#     if 'edge_color' not in visual_style.keys():
#         visual_style['edge_color'] = [edg_color_map[t] for t in g.es['steiner'] ]
        
#     if 'layout' not in visual_style.keys():
#         visual_style['layout'] = [(x[0],-x[1]) for x in g.vs['name']]
        
#     # if 'vertex_label' not in visual_style.keys():
#     #     visual_style['vertex_label'] = [str(x) for x in g.vs['name']]

#     return igraph.plot(g, output_file, **visual_style)

# def write_reblock_svg(g, output_file):
#     vtx_color_map = {True: 'red', False: 'blue'}
#     edg_color_map = {True: 'red', False: 'blue'}
    
#     visual_style = {}
#     if 'colors' not in visual_style.keys():
#         visual_style['colors'] = [vtx_color_map[t] for t in g.vs['terminal'] ]
    
#     BIG = 5
#     SMALL = 1

#     visual_style['width'] = 600
#     visual_style['height'] = 600

#     if 'vertex_size' not in visual_style.keys():
#         visual_style['vertex_size'] = [BIG if v['terminal'] else SMALL for v in g.vs]

#     if 'edge_colors' not in visual_style.keys():
#         visual_style['edge_colors'] = [edg_color_map[t] for t in g.es['steiner'] ]
        
#     if 'layout' not in visual_style.keys():
#         visual_style['layout'] = [(x[0],-x[1]) for x in g.vs['name']]
        
#     # if 'vertex_label' not in visual_style.keys():
#     #     visual_style['vertex_label'] = [str(x) for x in g.vs['name']]

#     g.write_svg(output_file, **visual_style)

