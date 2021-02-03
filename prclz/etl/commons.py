from pathlib import Path
from typing import Dict, Optional, Sequence

from urlpath import URL 
import requests

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

def download(src: URL, dst: Path) -> None:
    r = requests.get(src, stream = True)
    with dst.open('wb') as fd:
        for content in r.iter_content(chunk_size = 512):
            fd.write(content)