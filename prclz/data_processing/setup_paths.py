from pathlib import Path 
from typing import Dict 
import pandas as pd 

PRCLZ_ROOT = Path(__file__).parent.parent.absolute()
TRANS_TABLE = pd.read_csv(str(PRCLZ_ROOT / "data_processing" / "country_codes.csv"))

def build_data_dir(data_root: str) -> Dict:
    '''
    Builds data directory structure
    '''
    data_root = Path(data_root)
    geofabrik = data_root / 'input'
    geojson = data_root / 'geojson'
    blocks = data_root / 'blocks'
    bldgs = data_root / 'buildings'
    parcels = data_root / 'parcels'
    lines = data_root / 'lines'
    complexity = data_root / 'complexity'
    gadm = data_root / 'GADM'
    gadm_geojson = data_root / 'geojson_gadm'

    data_paths = {
      'root': data_root,
      'geofabrik': geofabrik,
      'geojson': geojson,
      'blocks': blocks,
      'bldgs': bldgs,
      'parcels': parcels,
      'lines': lines,
      'complexity': complexity,
      'gadm': gadm,
      'gadm_geojson': gadm_geojson
    }
    for v in data_paths.values():
        v.mkdir(parents=True, exist_ok=True)

    return data_paths

def get_example_paths() -> Dict:

    return build_data_dir( (PRCLZ_ROOT / "data") )

# # Path to main directory, will need to be set
# ROOT = Path("/home/cooper/Documents/chicago_urban/mnp/cooper_prclz")

# # Below this, paths will be automatically set based on ROOT
# DATA_PATH = ROOT / "data"
# GEOFABRIK_PATH = DATA_PATH / "input"
# GEOJSON_PATH = DATA_PATH / "geojson"   

# BLOCK_PATH = DATA_PATH / "blocks"
# BLDGS_PATH = DATA_PATH / "buildings"
# PARCELS_PATH = DATA_PATH / "parcels"
# LINES_PATH = DATA_PATH / "lines"
# COMPLEXITY_PATH = DATA_PATH / "complexity"

# GADM_PATH = DATA_PATH / "GADM"
# GADM_GEOJSON_PATH = DATA_PATH / "geojson_gadm"

# TRANS_TABLE = pd.read_csv((ROOT / "data_processing" / 'country_codes.csv'))

# all_paths = [BLOCK_PATH, GEOJSON_PATH, GADM_PATH, GADM_GEOJSON_PATH, 
#             PARCELS_PATH, GEOFABRIK_PATH, COMPLEXITY_PATH, BLDGS_PATH]

# # Create data dirs
# for p in all_paths:
#     p.mkdir(parents=True, exist_ok=True)
    
