from pathlib import Path
from typing import Dict, Optional, Sequence


def build_data_dir(root: str, additional: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    '''
    Builds data directory structure
    '''
    root = Path(root)
    data_paths = {
        folder: root/folder for folder in 
        ['input','geojson','blocks','buildings','parcels','lines','complexity','GADM','geojson_gadm'] + 
        (additional if additional else [])
    }
    data_paths["root"] = root       

    for v in data_paths.values():
        v.mkdir(parents=True, exist_ok=True)

    return data_paths
