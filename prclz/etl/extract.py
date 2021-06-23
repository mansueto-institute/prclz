from logging import error
from pathlib import Path
from subprocess import run

def _extract(output_dir: Path, pbf_path: Path, prefix_name: str, overwrite: bool):
    """Call the extract script to extract a pbf"""
    output_files = ['lines', 'building_linestrings', 'building_polygons']
    print([Path(output_dir + prefix_name + output_file + '.geojson').exists() for output_file in output_files])
    print(output_dir, pbf_path)
    if any([Path(output_dir + prefix_name + output_file + '.geojson').exists() for output_file in output_files]) and not overwrite:
        error(f"The files to be extracted already exist in {str(output_dir)} and overwrite is set to False")
    extract_path = Path(Path(__file__).resolve().parent) / 'scripts'
    run(['cd', extract_path, '&&', './extract.sh', pbf_path, (str(output_dir)+prefix_name)])