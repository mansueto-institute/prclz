from functools import reduce
from itertools import chain, combinations, permutations
from typing import List, Sequence, Tuple, Union, Type, Set, Callable
from logging import debug, info, warning, error
from pathlib import Path 
import geopandas as gpd
import pandas as pd
import geopy.distance as gpy
import igraph
import numpy as np
import tqdm
from shapely.geometry import (LinearRing, LineString, MultiLineString,
                              MultiPoint, MultiPolygon, Point, Polygon)
from shapely.ops import cascaded_union, unary_union, linemerge
from shapely.wkt import loads

BUF_EPS = 1e-4
BUF_RATE = 2
PlanarGraph = None

def edge_seq_to_linestring(e_seq, graph):

    lines = []
    for e in e_seq:
        e_line = LineString(graph.edge_to_coords(e))
        lines.append(e_line)
    return unary_union(lines)

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

def calc_metric_closure(G: PlanarGraph,
                   terminal_vertices: Sequence[igraph.Vertex],
                   ) -> PlanarGraph:
    # Build closed graph of terminal_vertices where each weight is the shortest path distance
    H = PlanarGraph()
    for u,v in combinations(terminal_vertices, 2):
        path_idxs = G.get_shortest_paths(u, v, weights='weight', output='epath')
        path_edges = G.es[path_idxs[0]]
        path_distance = reduce(lambda x,y : x+y, map(lambda x: x['weight'], path_edges))
        kwargs = {'weight':path_distance, 'path':path_idxs[0]}
        H.add_edge(u['name'], v['name'], **kwargs)
    return H

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

def distance_meters(a0, a1):
    """Helper for getting spatial dist from lon/lat"""

    lonlat_a0 = gpy.lonlat(*a0)
    lonlat_a1 = gpy.lonlat(*a1)

    return gpy.distance(lonlat_a0, lonlat_a1).meters

def distance(a0: Union[Sequence[float], np.ndarray], 
             a1: Union[Sequence[float], np.ndarray],
             ) -> float:
    """Flexible distance function for points, tuples, arrays"""
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simplified = False

    @classmethod
    def from_edges(cls: Type, 
                   edges: Sequence[Tuple[float, float]],
                   ) -> PlanarGraph:
        graph = cls()
        for edge in edges:
            graph.add_edge(*edge)
        return graph

    @classmethod
    def from_multipolygon(cls: Type, 
                          multipolygon: MultiPolygon,
                          ) -> PlanarGraph:
        """
        Factory method for creating a graph representation of 
        a multipolygon

        Args:
            multipolygon: shapely geometry

        Returns:   
            Creates a PlanarGraph instance
        """
        pgraph = cls()

        for parcel_id, polygon in enumerate(multipolygon):
            nodes = polygon.exterior.coords 
            for i, n in enumerate(nodes):
                if i != 0:
                    pgraph.add_edge(n, prev_n, parcel_id=parcel_id)
                prev_n = n 
        return pgraph 


    @classmethod
    def from_multilinestring(cls: Type, 
                             multilinestring: MultiLineString,
                             ) -> PlanarGraph:
        """
        Factory method for creating a graph representation of 
        a multilinestring

        Args:
            multilinestring: shapely geometry

        Returns:   
            Creates a PlanarGraph instance
        """
        pgraph = cls()

        for linestring in multilinestring:
            nodes = list(linestring.coords)
            for i, n in enumerate(nodes):
                if i!=0:
                    pgraph.add_edge(n, nodes[i-1])

        return pgraph

    def add_node(self, 
                 coords: Tuple[float, float], 
                 terminal: bool=False,
                 ) -> None:
        """
        Safely adds node, represented by coord pair, to the graph, 
        if needed. Allows for denoting nodes as terminal

        Args:
            coords: coordinates of node
            terminal: if True, node0 will be a target in Steiner Tree approx

        Returns:
            Modifies graph in-place, no return value
        """
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

    def add_edge(self, 
                 coords0: Tuple[float, float], 
                 coords1: Tuple[float, float], 
                 terminal0: bool=False, 
                 terminal1: bool=False, 
                 parcel_id: str=None, 
                 **kwargs,
                 ) -> None:
        """
        Safely adds edge, represented by two coord pairs, to the graph, 
        adding nodes as needed. Allows for denoting nodes as terminal
        and for storing the parcel_id

        Args:
            coords0: coordinates of first node in edge
            coords1: coordinates of second node in edge
            terminal0: if True, node0 will be a target in Steiner Tree approx
            terminal1: if True, node0 will be a target in Steiner Tree approx
            kwargs: Other attrs to pass to edge

        Returns:
            Modifies graph in-place, no return value
        """
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


    def split_edge_by_node(self, 
        edge_tuple: Tuple[Tuple[float, float], Tuple[float, float]], 
        coords: Tuple[float, float],
        terminal: bool=False,
        ) -> None:
        """
        Given an existing edge btwn 2 nodes, and a third unconnected node, 
        replaces the existing edge with 2 new edges with the previously
        unconnected node between the two
        NOTE: if the new node is already one of the edges, we do not create a self-edge

        Args:
            edge_tuple: edge, represented as a sequence of coords
            coords: coordinates of node

        Returns:
            Modifies the graph in-place, no return value
        """
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
    def closest_point_to_node(
        edge_tuple: Tuple[Tuple[float, float], Tuple[float, float]], 
        coords: Tuple[float, float],
        ) -> Tuple[float, float]:
        """
        Given a tuple repr of an edge and a coordinate, finds
        the nearest point on the edge to the coordinate

        NOTE: maybe should replace this with shapely.ops.nearest_points

        Args:
            edge_tuple: edge, represented as a sequence of coords
            coords: coordinates of node

        Returns:
            Coordinate pair of closest point
        """
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

    def coord_path_to_edge_path(self, 
                                coord_seq: List[Tuple[float, float]],
                                ) -> List[igraph.Edge]:
        '''
        Converts a path specified by coordinates to an edge sequence
        '''
        edge_list = []
        for c0, c1 in zip(coord_seq, coord_seq[1:]):
            edge = self.coords_to_edge(c0, c1)[0]
            edge_list.append(edge)
        return edge_list 

    def coords_to_edge(self, 
                       coord0: Tuple[float, float], 
                       coord1: Tuple[float, float],
                       ) -> igraph.EdgeSeq:
        """
        If we have the coordinates of two nodes, returns
        the edge 
        """
        v0 = self.vs.select(name=coord0)
        v1 = self.vs.select(name=coord1)
        edge = self.es.select(_between=(v0, v1))
        return edge 

    def edge_to_coords(self, 
                       edge: igraph.Edge, 
                       expand: bool=False,
                       ) -> Sequence[Tuple[float, float]]:
        """
        Converts a graph edge to a sequence of coordinates
        representation. Some graphs have been simplified, 
        meaning continuous edge sequences have been reduced
        to a single edge -- if expand is True, then unpacks 
        the simplification via the 'path' edge attr 

        Args:
            edge: graph edge to convert
            expand: if True, unpacks the 'path' attr and 
                    adds to the coordinate sequence

        Returns:
            Sequence of coordinate pairs
        """

        v0_idx, v1_idx = edge.tuple 
        v0_coords = self.vs[v0_idx]['name']
        v1_coords = self.vs[v1_idx]['name']
        if expand:
            edge_tuple = [v0_coords] + edge['path'] + [v1_coords]
        else:
            edge_tuple = (v0_coords, v1_coords)

        return edge_tuple 

    def _setup_linestring_attr(self) -> None:
        """
        Adds a Shapely linestring attr to edges which allows for 
        downstream filtering of edges via shapely binary predicates

        Returns:
            Modifies graph in-place, no return value
        """
        if 'linestring' not in self.es.attributes():
            self.es['linestring'] = [LineString(self.edge_to_coords(e)) for e in self.es]
        else:
            no_linestring_attr = self.es.select(linestring_eq=None)
            no_linestring_attr['linestring'] = [LineString(self.edge_to_coords(e)) for e in no_linestring_attr]

    def _cleanup_linestring_attr(self):
        """Removes linestring attr once we are done modifying graph"""
        del self.es['linestring']

    def _find_candidate_edges(self, 
                              coords: Tuple[float, float],
                              ) -> Sequence[igraph.Edge]:
        """
        Iteratively buffers around a proposed coord until intersecting
        edges are found. On large graphs, this can speed up the process
        of adding a node to the closest edge.

        Args:
            coords: coordinates of node
        
        Returns:
            Edges found to be within some buffer of the node
        """
        self._setup_linestring_attr()

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
        debug("Found %s/%s possible edges thru %s tries", len(edges), len(self.es), i)
        return edges 

    def add_node_to_closest_edge(self, 
                                 coords: Tuple[float, float], 
                                 terminal: bool=False, 
                                 fast: bool=True, 
                                 get_edge: bool=False,
                                 ) -> None:
        """
        Given a coordinate repr of a node, finds the closest edge
        on the graph, and extracts the closest point along that edge.
        Splits the edge at that closest point, creating two new edges
        to replace the single edge. 
        NOTE: This is used in reblocking when the building centroids, which
              lie in the 'face' of the graph, are moved to the closest
              edge-point and added to the graph

        Inputs:
            coords: coordinates of node
            terminal: if True, new node is target in Steiner Tree approx
            fast: if True, first finds candidate edges

        Returns:
            Modifies graph in-place, no return value
        """
        closest_edge_nodes = []
        closest_edge_distances = []

        if fast:
            cand_edges = self._find_candidate_edges(coords)
        else:
            cand_edges = self.es 

        for edge in cand_edges:
            edge_tuple = self.edge_to_coords(edge)

            #Skip self-edges
            if edge.is_loop():
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

    def add_buildings(self: PlanarGraph, 
                      bldg_centroids: List[Point],
                      ) -> PlanarGraph:
        """Add bldg centroids to planar graph"""

        total_blgds = len(bldg_centroids)
        for i, bldg_node in enumerate(bldg_centroids):
            self.add_node_to_closest_edge(bldg_node, terminal=True)

        if total_blgds > 0:
            self._cleanup_linestring_attr()
        #return graph 

    def clean_graph(self) -> PlanarGraph:
        """
        Some graphs are malformed bc of bad geometries and 
        we take the largest connected subgraph
        """
        is_conn = self.is_connected()
        if is_conn:
            num_components = 1
        else:
            components = self.components(mode=igraph.WEAK)
            num_components = len(components)
            comp_sizes = [len(idxs) for idxs in components]
            arg_max = np.argmax(comp_sizes)
            comp_indices = components[arg_max]
        
            self = self.subgraph(comp_indices)
        info("Graph contains %s components", num_components)
        return num_components

    def update_edge_types(self, 
                          block_polygon: Polygon, 
                          check: bool=False, 
                          lines_pgraph: PlanarGraph=None,
                          ) -> Tuple[int, int]:
        """
        When reblocking, existing roads receive 0 weight. Existing roads
        are captured in the block_polygon, so given a graph and a 
        block_geometry, this updates the edge types of the graph as
        either existing / new. Then it updates the weights.
        Graph is updated in place.

        Args:
            self: graph repr of parcel boundaries
            block_geom: Polygon of block geometry
            check: If true, verify that each point in the block is in fact in the parcel
            lines_pgraph: NA - will be deprecated

        Returns:
            Updates graph in place. Returns summary of matching
            check between parcel and block
        """
        block_coords_list = list(block_polygon.exterior.coords)
        coords = set(block_coords_list)

        rv = (None, None)
        missing = None
        total = None
        
        # Option to verify that each point in the block is in fact in the parcel
        if check:
            parcel_coords = set(v['name'] for v in self.vs)
            total = 0
            is_in = 0
            for coord in coords:
                is_in = is_in+1 if coord in parcel_coords else is_in 
                total += 1
            missing = total-is_in
            if missing != 0:
                warning("%s of %s block coords are NOT in the parcel coords", missing, total)

        # Get list of coord_tuples from the polygon
        assert block_coords_list[0] == block_coords_list[-1], "Not a complete linear ring for polygon"

        # Loop over the block coords (as define an edge) and update the corresponding edge type in the graph accordingly
        # NOTE: every block coord will be within the parcel graph vertices
        for i, n0 in enumerate(block_coords_list):
            if i==0:
                continue
            else:
                n1 = block_coords_list[i-1]
                u_list = self.vs.select(name_eq=n0)
                v_list = self.vs.select(name_eq=n1)
                if len(u_list) > 0 and len(v_list) > 0:
                    u = u_list[0]
                    v = v_list[0]
                    path_idxs = self.get_shortest_paths(u, v, weights='weight', output='epath')[0]

                    # the coords u and v from the block are
                    if lines_pgraph is None:
                        self.es[path_idxs]['edge_type'] = 'highway'
                    else:
                        pass 

        self.es.select(edge_type_eq='highway')['weight'] = 0

        rv = (missing, total)
        return rv 


    def calc_steiner_tree(self, 
                            terminal_vertices: Sequence[igraph.Vertex], 
                            ) -> Sequence[int]:
        """
        Steiner Tree approx on the graph G, where the terminal nodes are
        determined by terminal_vertices. The graph must have a 'weight'
        edge attr, which in the PlanarGraph class will default to the
        Euclidean distance

        Args:
            terminal_vertices: target vertices

        Returns:
            Indices of the edges included in the Steiner approx
        """
        H = calc_metric_closure(self, terminal_vertices)

        # Now get the MST of that complete graph of only terminal_vertices
        if "weight" not in H.es.attributes():
            error("----H graph does not have weight, ERROR There are %s", len(terminal_vertices))
        mst_edge_idxs = H.spanning_tree(weights='weight', return_tree=False)

        # Now, we join the paths for all the mst_edge_idxs
        steiner_edge_idxs = list(chain.from_iterable(H.es[i]['path'] for i in mst_edge_idxs))

        return steiner_edge_idxs, H

    def steiner_tree_approx(self, verbose: bool=False) -> None:
        """
        All Nodes within the graph have an attribute, Node.terminal, which is a boolean
        denoting whether they should be included in the set of terminal_nodes which
        are connected by the Steiner Tree approximation. Runs basic Steiner
        approx and then sets Node.steiner to True to indicate that an
        edge is included in the set of optimal Steiner paths

        Args:
            verbose: if True, prints details on steiner approximation

        Returns:
            Modifies graph in-place, no return value
        """
        terminal_nodes = self.vs.select(terminal_eq=True)

        steiner_edge_idxs, metric_closure = self.calc_steiner_tree(terminal_nodes)
        for i in steiner_edge_idxs:
            self.es[i]['steiner'] = True 

    def add_through_lines(self,
        top_k: int=None,
        ratio_cutoff: float=None,
        cost_fn: Callable[[igraph.Edge], float]=None,
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
        # Recalc edge weights, but save old weights to restore
        orig_weights = self.es['weight'][:]
        if cost_fn is None:
            cost_fn = lambda e: e['eucl_dist']/e['width']
        self.calc_edge_weight(cost_fn=cost_fn)

        terminal_vertices = self.vs.select(terminal_eq=True)
        orig_metric_closure = calc_metric_closure(self, terminal_vertices)
        steiner_edges = self.es.select(steiner_eq=True)
        
        opt_subgraph = self.subgraph_edges(steiner_edges)
        opt_metric_closure = calc_metric_closure(opt_subgraph, 
                                                 opt_subgraph.vs.select(terminal_eq=True),
                                                 )

        # Get ratio of new/orig shortest path distance
        combs_list = list(combinations(orig_metric_closure.vs, 2))
        for v0, v1 in combs_list:
            e_orig = orig_metric_closure.es.select(_within=[v0.index, v1.index])[0]
            e_opt = opt_metric_closure.es.select(_within=[v0.index, v1.index])[0]
            e_orig['ratio'] = e_opt['weight'] / e_orig['weight']

        # Get edges over a certain threshold, and add the 
        #     paths from the original metric closure to our reblock data
        post_process_lines = []
        self.es['is_through_line'] = False

        if top_k is not None:
            info("Selecting the top-%s ratios", top_k)
            ratios = orig_metric_closure.es['ratio'][:]
            ranking = np.argsort(ratios)[::-1][:int(top_k)]
            for idx in ranking:
                e = orig_metric_closure.es[idx]
                edge_path = self.es[e['path']]
                edge_path['is_through_line'] = True 
                path_linestring = edge_seq_to_linestring(edge_path, self)
                post_process_lines.append(path_linestring)

        elif ratio_cutoff is not None:
            info("Adding thru lines, ratios are: %s", orig_metric_closure.es['ratio'])
            for e in orig_metric_closure.es.select(ratio_gt = ratio_cutoff):
                edge_path = self.es[e['path']]
                edge_path['is_through_line'] = True 
                path_linestring = edge_seq_to_linestring(edge_path, self)
                post_process_lines.append(path_linestring)

        # Now reset to the original edge weights
        self.es['weight'] = orig_weights
        return post_process_lines


    def plot_reblock(self, output_file, visual_style=None):

        output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)

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

        print(output_file)
        igraph.plot(self, str(output_file), **visual_style)

    def get_steiner_linestrings(self, 
        expand: bool=True,
        return_polys: bool=False,
        edge_seq: igraph.EdgeSeq=None,
        ) -> Sequence[Union[MultiLineString, MultiPolygon]]:
        """
        After running steiner_tree_approx the graph edges have
        the Edge.steiner attribute, so extract all steiner edges, 
        splitting between existsing and new lines. 

        Args:
            expand: if True, unpacks the 'path' attr and 
                    adds to the coordinate sequence
            return_polys: if True, returns opt paths as Polygons.
                          if False (default), returns as Linestrings
        Returns:
            Shapely geometry of steiner paths, split btwn new 
            and existing roads
        """
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
            if isinstance(new_lines, LineString):
                new_lines = MultiLineString([new_lines])
            if isinstance(existing_lines, LineString):
                existing_lines = MultiLineString([existing_lines])

            return new_lines, existing_lines

    def get_terminal_points(self) -> MultiPoint:
        """
        Takes all the terminal nodes (ie buildings) and returns them as 
        shapely MultiPoint

        Returns:
            geometry of the terminal nodes
        """
        points = [Point(v['name']) for v in self.vs if v['terminal']]
        multi_point = unary_union(points)
        return multi_point

    def _simplify_node(self, vertex: igraph.Vertex) -> None:
        """
        If we simplify node B with connections A -- B -- C
        then we end up with (AB) -- C where the weight 
        of the edge between (AB) and C equals the sum of the
        weights between A-B and B-C

        NOTE: this allows the graph to simplify long strings of nodes

        Args:
            vertex: node B in the description, to be merged 
                    with its neighbor
        Returns:
            Modifies graph in-place, no return value
        """
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

    def simplify(self) -> None:
        """
        Many nodes exist to approximate curves in physical space. Calling 
        this collapses those nodes to allow for faster downstream 
        computation. The collapsed node path is stored in the edge
        attr 'path' and can be recovered via the 'expand' option in
        many downstream tasks

        Returns:
            Modifies graph in-place, no return
        """
        if 'path' not in self.es.attributes():
            self.es['path'] = [ [] for v in self.vs]

        for v in self.vs:
            num_neighbors = len(v.neighbors())
            if num_neighbors == 2 and not v['terminal']:
                self._simplify_node(v)
        self.simplified = True

    def to_pieces(self) -> List[List[int]]:
        '''
        Creates representation of graph 
        '''

        piece_list = []
        all_visited_idxs = set()

        info("Breaking to pieces...")
        for v in tqdm.tqdm(self.vs):
            if v.index not in all_visited_idxs:
                neighbors = v.neighbors()
                num_neighbors = len(neighbors)
                if num_neighbors == 2:
                    cont_vs = self._search_continuous_edge(v)
                    for v in cont_vs:
                        all_visited_idxs.add(v)
                    piece_list.append(cont_vs)
        return piece_list 

    ################################################### 
    ## ADDITIONAL ATTRIBUTES FOR ADVANCED REBLOCKING ##
    def _search_continuous_edge(self, 
                                v: igraph.Vertex, 
                                visited_indices: Set[int]=None,
                                ) -> Set[int]:
        """
        Given a vertex, v, will find the continuous string of
        vertices v is part of. 
        NOTE: Recursive function

        Args:
            v: current Vertex. If v has 2 neighbors, it's part
               of a continuous string. If not, recursion ends
            visited_indices:
        """
        if visited_indices is None:
            #visited_indices = []
            visited_indices = set()
        
        neighbors = v.neighbors()
        #visited_indices.append(v.index)
        visited_indices.add(v.index)

        if len(neighbors) != 2:
            return visited_indices
        else:
            for n in neighbors:
                if n.index not in visited_indices:
                    path = self._search_continuous_edge(n, visited_indices)
            return visited_indices 

    def _simplify_edge_width(self) -> None:
        """
        Resets edge width to be the min over
        a continuous segment. Sets width to be the min over continous
        paths. So given subgraph w/ Nodes A-B-C-D, the width
        is the min(AB,BC,CD). Wrt road segments, this is more
        reflective of real life where road width is constant
        between intersections
        """
        all_visited = set()

        debug("Simplifying edge width...")
        for v in self.vs:
            neighbors = v.neighbors()
            num_neighbors = len(neighbors)
            if num_neighbors == 2:
                # get the edge sequence
                cont_vs = self._search_continuous_edge(v)
                
                debug("\nVertex: %s", v['name'])
                debug([self.vs[i]['name'] for i in cont_vs])
                
                edges = self.es.select(_within=cont_vs)
                debug("Found %s edges", len(edges))

                min_width = min([e['width'] for e in edges])
                if min_width is None:
                    debug("Vertex %s has %s neighbors", v.index, num_neighbors)
                    debug("\tSegment is %s", cont_vs)
                for e in edges:
                    e['width'] = min_width

    def set_edge_width(self, 
                       other_geoms: List[Polygon],
                       simplify: bool=True,
                       ) -> None:
        """
        For each Edge, sets a 'width' attribute, which is the minimum
        distance from that Edge to a set of Polygons, provided in 
        other_geoms. Functionally, this captures the possible width of
        the road segment built along the edge. 

        Args:
            other_geoms: Edge width is calculated relative these geometries
            simplify: if True, the width is set to be the min over continous
                      paths. So given subgraph w/ Nodes A-B-C-D, the width
                      is the min(AB,BC,CD). Wrt road segments, this is more
                      reflective of real life where road width is constant
                      between intersections

        Returns:
            Modifies graph in-place, no return value
        """
        for e in self.es:
            e_line = LineString(self.edge_to_coords(e))
            distances = [e_line.distance(g) for g in other_geoms]
            e['width'] = min(distances) 
        if simplify:
            self._simplify_edge_width() 

    def calc_edge_weight(self,
                         cost_fn: Callable[[igraph.Edge], float]=None,
                         use_edge_type: bool=True,
                         ) -> None:
        """
        Recalculates the Edge 'weight' attribute based on the other
        Edge attrs. By default, the cost function will be:
            
            default_fn(E): E['eucl_dist']/E['width']) * (not E['edge_type']=='highway')

        However, user can specify a cost_fn which will be called on each
        edge, parsing the edge attrs accordingly. 

        Args:
            cost_fn: A user-defined function taking in an edge as input, 
                     and returns weight based on the edge attrs
            use_edge_type: changes the default cost_fn. If True, the
                           default cost_fn assigns zero weight to existing
                           roads. If False, edge_type has no affect
                           Used internally for adding_through_lines, when
                           the distinction btwn edge types is no longer valid

        Returns:
            Modifies graph in-place, no return value
        """
        # Stand-in vals for use in default cost_fn
        has_edge_type = "edge_type" in self.es.attributes()
        has_width = "width" in self.es.attributes()
        if not has_width:
            self.es['width'] = 1
        if not has_edge_type:
            self.es['edge_type'] = [None]*len(self.es)
        
        if cost_fn is None:
            cost_fn = lambda e: (e['eucl_dist']/e['width']) * (not e['edge_type']=='highway')

        self.es['weight'] = [cost_fn(e) for e in self.es]

        # Clean up the stand-in vals, if applicable
        if not has_width and "width" in self.es.attributes():
            del self.es['width']
        if not has_edge_type and "edge_type" in self.es.attributes():
            del self.es['edge_type']

    def simplify_reblocked_graph(self):

        # Extract the optimal subgraph of new lines only
        def filter(edge) -> bool:
            if 'is_through_line' in edge.attributes():
                return ((edge['steiner'] or edge['is_through_line']) and (edge['edge_type'] != 'highway'))
            else:
                return (edge['steiner'] and (edge['edge_type'] != 'highway'))

        opt_subgraph = self.subgraph_edges(self.es.select(filter))
        idx_v_pieces = opt_subgraph.to_pieces()
        simplified_linestrings = []

        print("Simplifing linestrings...")
        for v_indices in tqdm.tqdm(idx_v_pieces, total=len(idx_v_pieces)):
            # Convert vertices -> edges -> linestrings
            v_seq = opt_subgraph.vs[v_indices]
            edges = opt_subgraph.es.select(_within=v_seq)

            # And fetch the buffer area based on the width across 
            # that linestring/edge-sequence  

            edge_linestring, _, admiss_region, _ = opt_subgraph.get_steiner_linestrings(expand=False, return_polys=True, edge_seq=edges)
            edge_linestring = linemerge(edge_linestring)

            simplified_linestring = simplify_linestring(edge_linestring, admiss_region)
            simplified_linestrings.append(simplified_linestring)
        return simplified_linestrings      


# def convert_to_lines(planar_graph) -> MultiLineString:
#     lines = [LineString(planar_graph.edge_to_coords(e)) for e in planar_graph.es]
#     multi_line = unary_union(lines)
#     return multi_line 


# def find_edge_from_coords(g, coord0, coord1):
#     '''
#     Given a pair of coordinates, checks whether the graph g
#     contains an edge between that coordinate pair
#     '''
#     v0 = g.vs.select(name_eq=coord0)
#     v1 = g.vs.select(name_eq=coord1)
#     if len(v0)==0 or len(v1)==0:
#         return None 
#     else:
#         edge = g.es.select(_between=(v0, v1))
#         if len(edge)==0:
#             return None 
#         else:
#             return edge[0]
