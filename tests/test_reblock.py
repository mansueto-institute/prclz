import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.wkt import loads
import unittest
from logging import basicConfig, debug, info

from prclz.reblock import planar_graph
#from prclz.reblock.planar_graph_utils import update_edge_types
# import reblock2
# from path_cost import FlexCost
'''
To-Do:
 - Do another round of refactoring
 - Get the simplify linestring part working, and tested
 - Build out the momepy parcelization pipeline, and connect
   that to our reblocking 
'''


def make_square(lower_left_pt, w=1):
    pts = []
    x,y = lower_left_pt
    i,j = 0,0   
    pts.append((x + w*i, y + w*j))

    i,j = 1,0   
    pts.append((x + w*i, y + w*j))

    i,j = 1,1   
    pts.append((x + w*i, y + w*j))

    i,j = 0,1   
    pts.append((x + w*i, y + w*j))
    pts.append(lower_left_pt)
    return LineString(pts)

# def make_grid(lower_left_pt, h=3, w=3, delta=1):
#     multi = []
#     for i in range(w):
#         for j in range(h):
#             multi.append(make_square((i,j), delta))
#     return multi 

def create_test_grid(n):
    multi = []
    for i in range(n):
        for j in range(n):
            if i==0 and j==1:
                triangle = LineString([(0,1),(1,1),(1,2),(0,1)])
                multi.append(triangle)
            else:
                multi.append(make_square((i,j), 1))
    return planar_graph.PlanarGraph.from_multilinestring(multi)

# def create_connected_grid():
#     linestrings = []
#     linestrings.extend(make_grid((0,0), h=3,w=3,delta=1))
#     linestrings.extend(make_grid((6,0), h=3,w=3,delta=1))
#     connector = LineString([(3,0), (4,0), (5,0), (6,0)])
#     linestrings.append(connector)
#     g = planar_graph.PlanarGraph.from_multilinestring(linestrings)
#     g.es['width'] = 5
#     i = 1
#     for e in g.es[[24, 25, 26]]:
#         e['width'] = i
#         i += 1
#     return g 

def make_grids_w_targets():
    grid = create_test_grid(2)
    block_polygon = Polygon([(0,0),(2,0),(2,2),(1,2),(0,1),(0,0)])
    points = [
        (2, 2),
        (0, 0.2),
        (1.8, 1),
        (0.8, 2),
    ]

    grid2 = create_test_grid(2) 
    grid3 = create_test_grid(2) 
    grid4 = grid 
    for i, pt in enumerate(points):
        if i < 2:
            grid2.add_node_to_closest_edge(pt, terminal=True)
            grid3.add_node_to_closest_edge(pt, terminal=True)
            grid4.add_node_to_closest_edge(pt, terminal=True)
        elif i < 3:
            grid3.add_node_to_closest_edge(pt, terminal=True)
            grid4.add_node_to_closest_edge(pt, terminal=True)
        elif i < 4:
            grid4.add_node_to_closest_edge(pt, terminal=True)
        #grid.add_node_to_closest_edge(pt, terminal=True)
    grids = {2: grid2, 3: grid3, 4: grid4}
    return grids, block_polygon 

class TestBasicSteinerApprox(unittest.TestCase):
    """
    Does the most basic possible test of the Steiner Approx
    in the PlanarGraph class. The make_grids_w_targets() method 
    builds a simple test graph and adds 2, 3, and 4 target
    (i.e. terminal) nodes which are the target of the Steiner Alg.
    This test uses Eucl dist as weights.
    Width is not factored in.
    Existing roads are not factored in.
    """
    def _make_data(self):
        grids, block_polygon = make_grids_w_targets()
        return grids, block_polygon

    def test_basic2pt(self):
        test_grids, _ = self._make_data()
        answer_basic2pt = {
            "LINESTRING (0 1, 1 2)",
            "LINESTRING (0 1, 0 0.2)",
            "LINESTRING (1 2, 2 2)",    
            }
        test_grids = test_grids
        grid = test_grids[2]
        grid.steiner_tree_approx()
        grid_steiner = grid.get_steiner_linestrings(expand=False)
        new_steiner_wkt = {s.wkt for s in grid_steiner[0]}
        self.assertEqual(new_steiner_wkt, answer_basic2pt)

    def test_basic3pt(self):
        test_grids, _ = self._make_data()
        answer_basic3pt = {
            "LINESTRING (1 1, 0 1)",
            "LINESTRING (2 1, 2 2)",
            "LINESTRING (0 1, 0 0.2)",
            "LINESTRING (1 1, 1.8 1)",
            "LINESTRING (2 1, 1.8 1)",
            }
        test_grids = test_grids
        grid = test_grids[3]
        grid.steiner_tree_approx()
        grid_steiner = grid.get_steiner_linestrings(expand=False)
        new_steiner_wkt = {s.wkt for s in grid_steiner[0]}
        self.assertEqual(new_steiner_wkt, answer_basic3pt)

    def test_basic4pt(self):
        test_grids, _ = self._make_data()
        answer_basic4pt = {
            "LINESTRING (2 1, 2 2)",
            "LINESTRING (1 2, 2 2)",
            "LINESTRING (0 1, 0 0.2)",
            "LINESTRING (2 1, 1.8 1)",
            "LINESTRING (0 1, 0.9 1.9)",
            "LINESTRING (1 2, 0.9 1.9)"
            }
        grid = test_grids[4]
        grid.steiner_tree_approx()
        grid_steiner = grid.get_steiner_linestrings(expand=False)
        new_steiner_wkt = {s.wkt for s in grid_steiner[0]}
        self.assertEqual(new_steiner_wkt, answer_basic4pt)

class TestExistingSteinerApprox(unittest.TestCase):
    """
    Existing roads have weight=0 when reblocking. This test
    uses the block polygon (the outside of the grid) to reweight
    the PlanarGraph edges before the Steiner Approx.
    Without the block reweighting, take the hypot but WITH
    reweighting, lowest cost path is around the perimeter.

    The test graph is a simple square with one diagonal. The diag
    is the opt path, but upon weighting the perim=0, bc they already
    exist, the opt path becomes traversing the perimiter.
    """
    SAVE = False
    def _make_data(self):
        multi = []
        multi.append(LineString([(0,0),(1,0),(1,1),(0,0)]))
        multi.append(LineString([(0,0),(1,1),(0,1),(0,0)]))
        graph = planar_graph.PlanarGraph.from_multilinestring(multi)
        
        block_polygon = Polygon(make_square((0,0), 1))
        
        graph.add_node_to_closest_edge((0,0), terminal=True)
        graph.add_node_to_closest_edge((1,1), terminal=True)

        return graph, block_polygon

    def test_no_block(self):
        graph, block_polygon = self._make_data()
        answer_new = "LINESTRING (0 0, 1 1)"
        answer_exist = "GEOMETRYCOLLECTION EMPTY"   
        
        graph.steiner_tree_approx()
        graph_steiner = graph.get_steiner_linestrings(expand=False)
        new_steiner = graph_steiner[0]
        exist_steiner = graph_steiner[1]
        self.assertTrue(new_steiner.equals(loads(answer_new)))
        self.assertTrue(exist_steiner.equals(loads(answer_exist)))

        if TestExistingSteinerApprox.SAVE:
            p = "./reblock_tests/test_no_block.png"
            graph.plot_reblock(p)

    def test_w_block(self):
        graph, block_polygon = self._make_data()
        answer_new = "GEOMETRYCOLLECTION EMPTY"   
        answer_exist0 = "MULTILINESTRING ((0 0, 0 1), (0 1, 1 1))"
        answer_exist1 = "MULTILINESTRING ((0 0, 1 0), (1 0, 1 1))"
        
        graph.update_edge_types(block_polygon, check=True)
        
        graph.steiner_tree_approx()
        graph_steiner = graph.get_steiner_linestrings(expand=False)
        new_steiner = graph_steiner[0]
        exist_steiner = graph_steiner[1]

        self.assertTrue(new_steiner.equals(loads(answer_new)))
        exist_cond = (exist_steiner.equals(loads(answer_exist0))
                      or exist_steiner.equals(loads(answer_exist1)))
        self.assertTrue(exist_cond)

        if TestExistingSteinerApprox.SAVE:
            p = "./reblock_tests/test_w_block.png"
            graph.plot_reblock(p)

class TestWidthSteinerApprox(unittest.TestCase):
    """
    Defualt Steiner uses eucl_dist as the 'weight' param,
    but advanced options allow for specifying weight=eucl_dist/width.
    In this test, by restricting the width of the hypotenous path
    along the diagonal of a square, the optimal path changes from the
    diagonal to along the perimter of the square

    NOTE: same basic test graph as TestExistingSteinerApprox
    """
    SAVE = False
    def _make_data(self):
        multi = []
        multi.append(LineString([(0,0),(1,0),(1,1),(0.5,0.5),(0,0)]))
        multi.append(LineString([(0,0),(0.5,0.5),(1,1),(0,1),(0,0)]))
        graph = planar_graph.PlanarGraph.from_multilinestring(multi)
        graph.add_node_to_closest_edge((0,0), terminal=True)
        graph.add_node_to_closest_edge((1,1), terminal=True)
        
        # A 'house' that straddles the opt path very closely
        eps = 0.00001
        p0 = Polygon([(0.5, 0.5+eps),(0.55,0.55+eps),
                      (0.55,0.55+2*eps), (0.5,0.5+2*eps),
                      ]) 

        return graph, [p0]

    def test_w_width(self):
        graph, bldg_polys = self._make_data()
        # NOTE: shapely geom compares set theoretic values
        #       so [0,0.5,1] == [0,1]
        answer_new0 = "MULTILINESTRING ((0 0, 0 1), (0 1, 1 1))"
        answer_new1 = "MULTILINESTRING ((0 0, 1 0), (1 0, 1 1))"
        answer_exist = "GEOMETRYCOLLECTION EMPTY"   
        
        # Update width
        graph.set_edge_width(bldg_polys, simplify=True)

        # Update weight
        graph.calc_edge_weight()
        
        graph.steiner_tree_approx()
        graph_steiner = graph.get_steiner_linestrings(expand=False)
        new_steiner = graph_steiner[0]
        exist_steiner = graph_steiner[1]

        new_cond = (new_steiner.equals(loads(answer_new0))
                      or new_steiner.equals(loads(answer_new1)))
        self.assertTrue(new_cond)
        self.assertTrue(exist_steiner.equals(loads(answer_exist)))
        #print("\n\nDONE!!!")

class TestAddingThruStreets(unittest.TestCase):
    """
    Because Steiner Alg leaves trees, we have the option to
    connect trees. Test the add_through_lines method.
    """
    SAVE = False
    def _make_data(self):
        multi = []

        multi.append(make_square((0,0), w=3))
        multi.append(LineString([(0,0),(0.9,0.9)]))
        multi.append(LineString([(2.1,2.1),(3,3)]))
        multi.append(LineString([(0.9,0.9),(2.1,2.1)]))
        graph = planar_graph.PlanarGraph.from_multilinestring(multi)

        graph.add_node_to_closest_edge((0.9,0.9), terminal=True)
        graph.add_node_to_closest_edge((2.1,2.1), terminal=True)
        graph.add_node_to_closest_edge((0,1), terminal=True)

        block_poly = Polygon(make_square((0,0), w=3))

        return graph, block_poly

    def test_no_thru(self):
        graph, block_poly = self._make_data()
        answer_new = "MULTILINESTRING ((0 0, 0.9 0.9), (2.1 2.1, 3 3))"
        answer_exist0 = "MULTILINESTRING ((0 0, 0 3), (0 3, 3 3))"
        answer_exist1 = "MULTILINESTRING ((3 3, 3 0), (3 0, 0 0))"

        graph.update_edge_types(block_poly, check=True)
        graph.steiner_tree_approx()
        graph_steiner = graph.get_steiner_linestrings(expand=False)
        new_steiner = graph_steiner[0]
        exist_steiner = graph_steiner[1]

        self.assertTrue(new_steiner.equals(loads(answer_new)))

        exist_cond = (exist_steiner.equals(loads(answer_exist0)) or
                      exist_steiner.equals(loads(answer_exist1)))
        self.assertTrue(exist_cond)

    def test_add_thru(self):
        graph, block_poly = self._make_data()
        answer_new = "MULTILINESTRING ((0 0, 3 3))"
        answer_exist0 = "MULTILINESTRING ((0 0, 0 3), (0 3, 3 3))"
        answer_exist1 = "MULTILINESTRING ((3 3, 3 0), (3 0, 0 0))"

        graph.update_edge_types(block_poly, check=True)
        graph.steiner_tree_approx()

        graph.add_through_lines(ratio_cutoff=2)

        graph_steiner = graph.get_steiner_linestrings(expand=False)
        new_steiner = graph_steiner[0]
        exist_steiner = graph_steiner[1]

        self.assertTrue(new_steiner.equals(loads(answer_new)))

        exist_cond = (exist_steiner.equals(loads(answer_exist0)) or
                      exist_steiner.equals(loads(answer_exist1)))
        self.assertTrue(exist_cond)



if __name__ == "__main__":
    basicConfig(level='DEBUG')
    unittest.main()


# # (1) Original reblocking
# test_grids = get_test_grids()
# for n, grid in test_grids.items():
#     grid.steiner_tree_approx()
#     p = "./reblock_tests/orig_n{}.png".format(n)
#     grid.plot_reblock(p)

# # (2) Flex reblocking but with default settings (should match #1)
# test_grids = get_test_grids()
# default_cost_fn = FlexCost()
# for n, grid in test_grids.items():
#     grid.flex_steiner_tree_approx(cost_fn = default_cost_fn)
#     p = "./test_dir/flex_baseline_n{}.png".format(n)
#     grid.plot_reblock(p)

# # (3) Flex reblocking -- penalize turns
# test_grids = get_test_grids()
# noturn_cost_fn = FlexCost(lambda_turn_angle=2.)
# for n, grid in test_grids.items():
#     grid.set_node_angles()
#     grid.flex_steiner_tree_approx(cost_fn = noturn_cost_fn)
#     p = "./test_dir/flex_no-turns_n{}.png".format(n)
#     grid.plot_reblock(p)

# # (4) Flex reblocking -- penalize non-4 way compatible intersections
# test_grids = get_test_grids()
# fourway_cost_fn = FlexCost(lambda_degree=200., lambda_turn_angle=2.)
# for n, grid in test_grids.items():
#     grid.set_node_angles()
#     grid.flex_steiner_tree_approx(cost_fn = fourway_cost_fn)
#     p = "./test_dir/flex_fourways_n{}.png".format(n)
#     grid.plot_reblock(p)

# # (5) Flex reblocking -- add 'houses' and factor in width
# test_grids = get_test_grids()
# ls = make_square(lower_left_pt=(0.5, 1.1), w=0.2)
# bldg = [Polygon(ls)]
# width_cost_fn = FlexCost(lambda_width=1.0)
# for n, grid in test_grids.items():
#     grid.set_edge_width(bldg)
#     grid.flex_steiner_tree_approx(cost_fn = width_cost_fn)
#     p = "./test_dir/flex_width_n{}.png".format(n)
#     grid.plot_reblock(p)



# region = 'Africa'
# gadm_code = 'DJI'
# gadm = 'DJI.1.1_1'
# block_list = ['DJI.1.1_1_1']

# # (6) Baseline Test on DJI data
# cost_fn = FlexCost()
# reblock_data, reblock_poly_data, parcels, buildings, blocks =  reblock2.reblock_gadm(region, gadm_code, 
#                                                                                     gadm, cost_fn, 
#                                                                                     block_list=block_list,
#                                                                                     return_metric_closures=False,
#                                                                                     return_planar_graphs=False)
# reblock_data = gpd.GeoDataFrame.from_dict(reblock_data)                                                                                   
# reblock_poly_data = gpd.GeoDataFrame.from_dict(reblock_poly_data)                                                                                  

# bldgs = gpd.read_file('../data/buildings/Africa/DJI/buildings_DJI.1.1_1.geojson')
# ax_baseline = plot_dji_test('DJI.1.1_1_1', 'dji_test.png')

# (7) Baseline + width
# cost_fn = FlexCost(lambda_width = 1.0)
# reblock_output =  reblock2.reblock_gadm(region, gadm_code, gadm, cost_fn, 
#                                         block_list=block_list,
#                                         return_metric_closures=True,
#                                         return_planar_graphs=True)
# reblock_data, reblock_poly_data, parcels, buildings, blocks, metric_closures, planar_graphs = reblock_output
# bldgs = gpd.read_file('../data/buildings/Africa/DJI/buildings_DJI.1.1_1.geojson')
# reblock_data = gpd.GeoDataFrame.from_dict(reblock_data)                                                                                   
# reblock_poly_data = gpd.GeoDataFrame.from_dict(reblock_poly_data)                                                                                  

# ax_width = plot_dji_test('DJI.1.1_1_1', 'dji_width_test.png')

# (8) Baseline + width + simplification
# cost_fn = FlexCost(lambda_width = 1.0)
# reblock_output =  reblock2.reblock_gadm(region, gadm_code, gadm, cost_fn, 
#                                         block_list=block_list,
#                                         return_metric_closures=True,
#                                         return_planar_graphs=True,
#                                         simplify_new_roads=True)
# reblock_data, reblock_poly_data, parcels, buildings, blocks, metric_closures, planar_graphs = reblock_output
# bldgs = gpd.read_file('../data/buildings/Africa/DJI/buildings_DJI.1.1_1.geojson')
# reblock_data = gpd.GeoDataFrame.from_dict(reblock_data)                                                                                   
# reblock_poly_data = gpd.GeoDataFrame.from_dict(reblock_poly_data)                                                                                  

# ax_width = plot_dji_test('DJI.1.1_1_1', 'dji_width_simpl_test.png')

# (9) Baseline + width + thru_streets + simplification
# cost_fn = FlexCost(lambda_width = 1.0)
# reblock_output =  reblock2.reblock_gadm(region, gadm_code, gadm, cost_fn, 
#                                         block_list=block_list,
#                                         return_metric_closures=True,
#                                         return_planar_graphs=True,
#                                         through_street_cutoff=0.7,
#                                         simplify_new_roads=True)
# reblock_data, reblock_poly_data, parcels, buildings, blocks, metric_closures, planar_graphs = reblock_output
# bldgs = gpd.read_file('../data/buildings/Africa/DJI/buildings_DJI.1.1_1.geojson')
# reblock_data = gpd.GeoDataFrame.from_dict(reblock_data)                                                                                   
# reblock_poly_data = gpd.GeoDataFrame.from_dict(reblock_poly_data)                                                                                  

# ax_width = plot_dji_test('DJI.1.1_1_1', 'dji_width_thrustr_simpl_test.png')


# def add_buildings(parcel_poly_df: gpd.GeoDataFrame, 
#                   building_list: List[Point],
#                   planar_graph: planar_graph.PlanarGraph,
#                   ):

#     def cost_fn(centroid: Point, edge: igraph.Edge) -> float:
#         dist = LineString(planar_graph.edge_to_coords(edge)).distance(centroid)
#         width = edge['width']
#         return dist + dist / width 

#     bldg_gdf = gpd.GeoDataFrame.from_dict({'geometry': building_list})
#     bldg_joined = gpd.sjoin(bldg_gdf, parcel_poly_df, how='left')

#     for _, (centroid, parcel_id) in bldg_joined[['geometry', 'parcel_id']].iterrows():
#         edge_seq = planar_graph.edges_in_parcel(parcel_id)
#         edge_cost = map(lambda e: cost_fn(centroid, e), edge_seq)
#         argmin = np.argmin(list(edge_cost))

#         closest_edge = edge_seq[argmin]
#         planar_graph.add_bldg_centroid(centroid, closest_edge)

# t_parcels, t_buildings, t_blocks = reblock2.load_reblock_inputs(region, gadm_code, gadm)

# # (1) Do the initial reblocking
# rv = reblock2.reblock_block_id(parcels, 
#                      buildings,
#                      blocks,
#                      block_id = block_list[0],
#                      cost_fn = cost_fn,
#                      return_metric_closure = True,
#                      reblock_data = None,
#                      reblock_poly_data = None)

# reblock_data, reblock_poly_data, planar_graph, metric_closure = rv

# # (2) Now add some through streets
# ratio_cutoff = 0.7
# through_lines = simplify_reblock.get_through_lines(planar_graph, metric_closure, 
#                                                    ratio_cutoff, cost_fn)
