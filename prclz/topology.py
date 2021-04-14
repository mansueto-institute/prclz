import pickle
from itertools import chain, product
from logging import debug
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rtree
from shapely.geometry import LineString, MultiLineString, Point, Polygon

MAX_CENTROID_DEGREE = 100 

""" Implementation of a Planar Graph """

class Node:
    """ two-dimensional point container """

    def __init__(self, coordinates, name=None):
        assert len(coordinates) == 2, "input coordinates must be of length 2"
        self.x, self.y = coordinates
        self.coordinates = (self.x, self.y)
        self.road = False
        self.interior = False
        self.barrier = False
        self.terminal = False    # This denotes whether we Node is the target of Steiner Tree Approx
        self.name = name

    @staticmethod
    def from_point(point: Point):
        '''
        Helper function to convert shapely.Point -> Node
        '''
        return Node(point.coords[0])

    def __repr__(self):
        return self.name if self.name else "Node(%.2f,%.2f)" % (self.x, self.y)

    def __eq__(self, other):
        if other is None:
            return False
        return self.coordinates == other.coordinates

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.coordinates < other.coordinates

    def __hash__(self):
        return hash(self.coordinates)

    def __getitem__(self, idx):
        return self.coordinates[idx]

    def __sub__(self, other):
        coords = (self.x - other.x, self.y - other.y)
        return Node(coords)

    def __add__(self, other):
        coords = (self.x + other.x, self.y + other.y)
        return Node(coords)

    def scalar_multiple(self, scalar):
        '''
        Returns self but scaled
        '''
        coords = (self.x * scalar, self.y * scalar)
        return Node(coords)

    def distance(self, other):
        return np.linalg.norm((self.x - other.x, self.y - other.y))


class Edge:
    """ undirected edge as a tuple of nodes,
    with flags to indicate if the edge ios interior, road, or barrier """

    def __init__(self, nodes: Sequence[Node]):
        # nodes = sorted(nodes, lambda p: (p.x, p.y))
        self.nodes = nodes
        self.interior = False
        self.road = False
        self.barrier = False

    def length(self):
        return self.nodes[0].distance(self.nodes[1])

    def min_distance_to_node(self, node):
        '''
        Just returns the min distance from the edge to the node
        '''
        x1,y1 = self.nodes[0]
        x2,y2 = self.nodes[1]
        x0,y0 = node 

        num = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        den = np.sqrt((y2-y1)**2 + (x2-x1)**2)

        return num/den 

    def vector_projection(self, node):
        '''
        Returns the vector projection of node onto the LINE defined
        by the edge
        https://en.wikipedia.org/wiki/Vector_projection
        https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        '''
        a_vector = np.array(node.coordinates)

        b_vector_node = self.nodes[0] - self.nodes[1]
        b_vector = np.array(b_vector_node.coordinates)
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

        #print("unit vector = {} | normal vector = {}".format(b_unit, b_normal))

        min_distance = self.min_distance_to_node(node)

        # Depending on the ordering the +/- can get reversed so this is 
        # just a little hacky workaround to make it 100% robust
        proj1 = a_vector + min_distance * b_normal
        proj2 = a_vector - min_distance * b_normal

        # The correct projection will min the new distance
        node_proj1 = Node(proj1)
        node_proj2 = Node(proj2)

        if np.abs(self.min_distance_to_node(node_proj1)) < 10e-4:
            return node_proj1
        elif np.abs(self.min_distance_to_node(node_proj2)) < 10e-4:
            return node_proj2
        else:
            assert False, "Vector projection failed"


    def node_on_edge(self, node):
        '''
        Because line segments are finite, when calculating min distance from edge
        to a point we need to check whether the projection onto the LINE defined by
        the edge is in fact on the edge or outside of it
        '''

        mid_x = (self.nodes[0][0]+self.nodes[1][0]) / 2.
        mid_y = (self.nodes[0][1]+self.nodes[1][1]) / 2.
        mid_node = Node((mid_x, mid_y))

        # NOTE: the distance from the midpoint of the edge to any point on the edge
        #       cannot be greater than the dist to the end points
        max_distance = mid_node.distance(self.nodes[0])
        assert np.abs(mid_node.distance(self.nodes[0]) - mid_node.distance(self.nodes[1])) < 10e-4, "NOT TRUE MIDPOINT"

        node_distance = mid_node.distance(node)

        if node_distance > max_distance:
            return False
        else:
            return True 

    def closest_point_to_node(self, node):
        '''
        Returns the closest point on the edge, to the given node
        '''

        projected_node = self.vector_projection(node)
        if self.node_on_edge(projected_node):
            return projected_node
        else:
            dist_node0 = self.nodes[0].distance(node)
            dist_node1 = self.nodes[1].distance(node)
            if dist_node0 <= dist_node1:
                return self.nodes[0]
            else:
                return self.nodes[1]

    def __str__(self):
        return "Edge(({}, {}), ({}, {}))".format(
            self.nodes[0].x, self.nodes[0].y, self.nodes[1].x, self.nodes[1].y
        )

    def __repr__(self):
        return "Edge(({}, {}), ({}, {}))".format(
            self.nodes[0].x, self.nodes[0].y, self.nodes[1].x, self.nodes[1].y
        )

    def __eq__(self, other):
        return (
            self.nodes[0] == other.nodes[0] and
            self.nodes[1] == other.nodes[1]) or (
            self.nodes[0] == other.nodes[1] and
            self.nodes[1] == other.nodes[0])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.nodes)


class Face:
    """ polygon defined by edges """

    def __init__(self, edges):
        # determine representation of edges: Edge class or tuple?
        if len(edges) > 0 and type(edges[0]) != tuple:
            node_set = set(chain.from_iterable(edge.nodes for edge in edges))
            self.edges = set(edges)
            self.ordered_edges = edges
        else:
            node_set = set(chain.from_iterable(edges))
            planar_edges = list(map(Edge, edges))
            self.edges = set(planar_edges)
            self.ordered_edges = planar_edges
        self.nodes = list(sorted(node_set))
        self.name = ".".join(map(str, self.nodes))
        self._centroid = None

    def area(self):
        return 0.5*abs(sum(e.nodes[0].x*e.nodes[1].y - e.nodes[1].x*e.nodes[0].y for e in self.ordered_edges))

    def bounds(self):
        nodes = iter(self.nodes)
        xmin, ymin = next(nodes)
        xmax, ymax = xmin, ymin
        for node in nodes:
            if node.x < xmin: xmin = node.x
            if node.x > xmax: xmax = node.x
            if node.y < ymin: ymin = node.y
            if node.y > ymax: ymax = node.y
        return (xmin, ymin, xmax, ymax)

    def centroid(self):
        """finds the centroid of a MyFace, based on the shoelace method
        e.g. http://en.wikipedia.org/wiki/Shoelace_formula and
        http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
        The method relies on properly ordered edges. """

        if self._centroid:
            return self._centroid
        # acc_a2 here is 2 * (the summand for area) in the wikipedia formula 
        acc_a2, acc_cx, acc_cy = 0, 0, 0
        for e in self.ordered_edges:
            acc_a2 += e.nodes[0].x * e.nodes[1].y - e.nodes[1].x * e.nodes[0].y
            acc_cx += (e.nodes[0].x + e.nodes[1].x) * (e.nodes[0].x*e.nodes[1].y - e.nodes[1].x*e.nodes[0].y)
            acc_cy += (e.nodes[0].y + e.nodes[1].y) * (e.nodes[0].x*e.nodes[1].y - e.nodes[1].x*e.nodes[0].y)
        if abs(acc_a2) < 0.02:
            cx, cy, n = 0, 0, len(self.nodes)
            for node in self.nodes:
                cx, cy = cx + node.x, cy + node.y
            cx, cy = cx/n, cy/n
        else:
            a6 = 3*acc_a2
            cx, cy = acc_cx/a6, acc_cy/a6

        self._centroid = Node((cx, cy))
        return self._centroid

    def __len__(self):
        return len(self.edges)


class PlanarGraph(nx.Graph):
    
    def __init__(self, name: str = "S", dual_order: int = 0, incoming_graph_data = None, **attr):
        attr["name"] = name
        attr["dual_order"] = dual_order
        super().__init__(incoming_graph_data = incoming_graph_data, **attr)

    # static constructors for all the various ways we generate planar graphs 
    @staticmethod
    def from_edges(edges, name="S"):
        graph = PlanarGraph(name=name)
        for edge in edges:
            graph.add_edge(edge)
        return graph

    @staticmethod
    def from_polygons(polygons: Sequence[Polygon], name="S"):
        n = len(polygons)
        debug("Building planar graph %s from %s polygons", name, n)
        nodes = dict()
        faces = []
        for (i, polygon) in enumerate(polygons):
            polygon_nodes = []
            for node in map(Node, polygon.exterior.coords):
                if node not in nodes:
                    polygon_nodes.append(node)
                    nodes[node] = node
                else:
                    polygon_nodes.append(nodes[node])
            edges = [(polygon_nodes[i], polygon_nodes[i+1]) for i in range(len(polygon_nodes)-1)]
            faces.append(Face(edges))
            debug("processed polygon %s; total number of faces: %s", i, len(faces))

        graph = PlanarGraph(name=name)

        for edge in chain.from_iterable(face.edges for face in faces):
            graph.add_edge(Edge(edge.nodes))

        return graph

    @staticmethod
    def from_linestring(linestring: LineString, append_connection:bool=True):
        '''
        Helper function to convert a single Shapely linestring
        to a PlanarGraph
        '''

        # linestring -> List[Nodes]
        nodes: Iterable[Node] = [Node(p) for p in linestring.coords]

        # List[Nodes] -> List[Edges]
        if append_connection:
            nodes.append(nodes[0])
        edges : Iterable[Edge] = []
        for i, n in enumerate(nodes):
            if i==0:
                continue
            else:
                edges.append(Edge((n, nodes[i-1])))

        return PlanarGraph.from_edges(edges)
    
    @staticmethod
    def from_multilinestring(multilinestring: MultiLineString):
        '''
        Helper function to convert a Shapely multilinestring
        to a PlanarGraph
        '''

        pgraph = PlanarGraph()

        for linestring in multilinestring:
            # linestring -> List[Nodes]
            nodes = [Node(p) for p in linestring.coords]

            # List[Nodes] -> List[Edges]
            nodes.append(nodes[0])
            for i, n in enumerate(nodes):
                if i==0:
                    continue
                else:
                    pgraph.add_edge(Edge((n, nodes[i-1])))

        return pgraph

    @staticmethod
    def from_file(file_path: str):
        '''
        Loads a planar graph from a saved via
        '''

        with open(file_path, 'rb') as file:
            graph = pickle.load(file)
        return graph

    def __repr__(self):
        return "{}{} with {} nodes".format(self.name, self.graph["dual_order"], self.number_of_nodes())

    def __str__(self):
        return self.__repr__()

    def add_edge(self, edge: Edge, weight=None):
        assert isinstance(edge, Edge)
        super().add_edge(
            edge.nodes[0],
            edge.nodes[1],
            planar_edge=edge,
            weight=weight if weight else edge.length(),
        )

    def split_edge_by_node(self, edge_tuple, node: Node, weight=None):
        '''
        Given an existing edge btwn 2 nodes, and a third unconnected node, 
        replaces the existing edge with 2 new edges with the previously
        unconnected node between the two
        NOTE: if the new node is already one of the edges, we do not create a self-edge
        '''
        orig_node0, orig_node1 = edge_tuple
        if node == orig_node0:
            orig_node0.terminal = node.terminal 
        elif node == orig_node1:
            orig_node1.terminal = node.terminal 
        else:
            super().remove_edge(orig_node0, orig_node1)

            new_edge0 = Edge([orig_node0, node])
            new_edge1 = Edge([orig_node1, node])
            self.add_edge(new_edge0)
            self.add_edge(new_edge1)


    def get_embedding(self):
        return {
            node: sorted(
                self.neighbors(node),
                key=lambda neighbor, node=node: np.arctan2(
                    neighbor.x - node.x,
                    neighbor.y - node.y)
            ) for node in self.nodes()
        }

    def trace_faces(self):
        """Algorithm from SAGE"""
        if len(self.nodes()) < 2:
            return []

        embedding = self.get_embedding()
        edgeset = set(chain.from_iterable([
            [(edge[0], edge[1]), (edge[1], edge[0])]
            for edge in self.edges()
        ]))

        # begin face tracing
        faces = []
        face = [edgeset.pop()]
        while edgeset:
            neighbors = embedding[face[-1][-1]]
            next_node = neighbors[(neighbors.index(face[-1][-2])+1) %
                                  (len(neighbors))]
            candidate_edge = (face[-1][-1], next_node)
            if candidate_edge == face[0]:
                faces.append(face)
                face = [edgeset.pop()]
            else:
                face.append(candidate_edge)
                edgeset.remove(candidate_edge)
        # append any faces under construction when edgeset exhausted
        if len(face) > 0:
            faces.append(face)

        # remove the outer "sphere" face
        facelist = sorted(faces, key=len)
        self.outerface = Face(facelist[-1])
        self.outerface.edges = [self[e[1]][e[0]]["planar_edge"]
                                for e in facelist[-1]]
        for face in facelist[:-1]:
            inner_face = Face(face)
            inner_face.edges = [self[e[1]][e[0]]["planar_edge"] for e in face]
            yield inner_face

        # return inner_facelist

    def weak_dual(self):
        dual = PlanarGraph(name = self.name, dual_order = self.graph["dual_order"] + 1)
        debug("building r tree for %s", self)
        idx = rtree.index.Index()
        for i, f in enumerate(self.trace_faces()):
            idx.insert(i, f.bounds(), f)
        
        debug("building weak dual for %s", self)
        for (fn, face1) in enumerate(self.trace_faces()):
            nearest = list(_.object for _ in idx.nearest(face1.bounds(), MAX_CENTROID_DEGREE, objects=True))
            debug("nearest-polygon search for face %s yielded %s results", fn, len(nearest))
            for face2 in nearest:
                edges1 = [e for e in face1.edges if not e.road]
                edges2 = [e for e in face2.edges if not e.road]
                linestrings1 = [LineString([(e.nodes[0].x, e.nodes[0].y), (e.nodes[1].x, e.nodes[1].y)]) for e in face1.edges]
                linestrings2 = [LineString([(e.nodes[0].x, e.nodes[0].y), (e.nodes[1].x, e.nodes[1].y)]) for e in face2.edges]
                if len(set(edges1).intersection(edges2)) > 0 or any((e1.intersects(e2) and e1.touches(e2) and e1.intersection(e2).type != "Point") for (e1, e2) in product(linestrings1, linestrings2)):
                    dual.add_edge(Edge((face1.centroid(), face2.centroid())))

        return dual

    def add_node_to_closest_edge(self, node):
        '''
        Given the input node, this finds the closest point on each edge to that input node.
        It then adds that closest node to the graph. It splits the argmin edge into two
        corresponding edges so the new node is fully connected
        '''
        closest_edge_nodes = []
        closest_edge_distances = []
        edge_list = list(self.edges)

        for edge_tuple in edge_list:

            #Skip self-edges
            if edge_tuple[0] == edge_tuple[1]:
                #print("\nSKIPPING EDGE BC ITS A SELF-EDGE\n")
                continue 
            edge = Edge(edge_tuple)
            closest_node = edge.closest_point_to_node(node)
            closest_distance = closest_node.distance(node)
            closest_edge_nodes.append(closest_node)
            closest_edge_distances.append(closest_distance)

        argmin = np.argmin(closest_edge_distances)
        closest_node = closest_edge_nodes[argmin]
        closest_edge = edge_list[argmin]

        # Set attributes
        closest_node.terminal = node.terminal 

        # Now add it
        self.split_edge_by_node(closest_edge, closest_node)

    def plot(self, **kwargs):
        plt.axes().set_aspect(aspect=1)
        plt.axis("off")
        edge_kwargs = kwargs.copy()
        nlocs = {node: (node.x, node.y) for node in self.nodes}
        edge_kwargs["label"] = "_nolegend"
        edge_kwargs["pos"] = nlocs
        edge_kwargs.pop("node_color")
        nx.draw_networkx_edges(self, **edge_kwargs)
        node_kwargs = kwargs.copy()
        node_kwargs.pop("width")
        node_kwargs.pop("edge_color")
        node_kwargs["label"] = self.name
        node_kwargs["pos"] = nlocs
        nodes = nx.draw_networkx_nodes(self, **node_kwargs)
        if nodes:
            nodes.set_edgecolor("None")

    def save(self, file_path):
        '''
        Saves planar graph to file via pickle 
        '''

        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
