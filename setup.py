import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prclz",
    version="0.9.0",
    author="Mansueto Institute for Urban Innovation",
    author_email="miurbandata@uchicago.edu",
    description="Code for analysis related to the Million Neighborhoods Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/mansueto-institute/prclz",
    packages=setuptools.find_packages(),
    package_data={"prclz": ["etl/country_codes.csv", 
                            "scripts/extrct.sh",
                            "scripts/osmconf.ini"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={ 
        "console_scripts": ["prclz=prclz.cli:prclz"]
    }, 
    install_requires=[
        "wheel",
        "setuptools",
        "Click<8",
        "rtree",
        "shapely",
        "geopandas",
        "geos",
        "momepy",
        "python-igraph",
        "networkx",
        "scipy",
        "pytess",
        "matplotlib",
        "psutil",
        "urlpath",
        "geopy"
    ]
)