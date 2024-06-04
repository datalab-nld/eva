import setuptools

PACKAGE_NAME = 'evametoc'
SOURCE_DIRECTORY = 'src'

setuptools.setup(
    name=PACKAGE_NAME,
    version="0.0.15",
    description="A package to encode/decode environmental NetCDF-files into sets of video files",
    author="Daan Gommers, Dennis Strik, Vincent van Leijen",
    author_email="datalabjivc@mindef.nl",
    python_requires=">=3.7",
    package_dir={'': SOURCE_DIRECTORY},
    packages=setuptools.find_packages(where=SOURCE_DIRECTORY),
    
    install_requires=[
        "python-dateutil>=2.8.2",
        "cftime>=1.6.2",
        "numpy>=1.21.4",
        "xarray>=2022.12.0",
        "netCDF4>=1.6.2",
        "pandas>=1.5.2",
        "ffmpeg-python>=0.2.0",
        "scipy>=1.10.0",
    ],
    license="EUPL License",
)
