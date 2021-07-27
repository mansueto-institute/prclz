from logging import error, info
from pathlib import Path
from subprocess import call

def extract(pbf_path: Path, output_dir: Path, overwrite: bool):
    """Call the extract script to extract a pbf"""
    output_files = ['lines', 'building_linestrings', 'building_polygons']
    output_dir   = Path(output_dir).resolve()
    if any([Path(str(Path(output_dir).resolve()) + '/' + pbf_path.stem.split('-latest')[0] + '_' + output_file + '.geojson').exists() for output_file in output_files]) and not overwrite:
        error(f"One or more of the files to be extracted already exist in {str(output_dir)} and overwrite is set to False")
        raise Exception
    extract_path = Path(Path(__file__).resolve().parent.parent) / 'scripts'
    output = call(['bash', 'extract.sh', pbf_path.resolve(), str(output_dir)+'/'], cwd=str(extract_path))
    if output == 0:
        info('extract.sh completed successfully')
    else:
        error('extract.sh failed')
        raise Exception

