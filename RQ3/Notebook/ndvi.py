import requests
from bs4 import BeautifulSoup
import os
import xarray as xr
import geopandas as gpd
import rasterio.features
import numpy as np
import pandas as pd
from shapely.geometry import mapping
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# from netCDF import Dataset

base_url = "https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/2023/"

download_dir = "ndvi_2023"
os.makedirs(download_dir,exist_ok=True)

response = requests.get(base_url)
soup = BeautifulSoup(response.text,'html.parser')

nc_files = [a['href'] for a in soup.find_all('a',href=True) if a['href'].endswith('.nc')]

for nc_file in nc_files:
    file_url = base_url + nc_file
    local_path = os.path.join(download_dir, nc_file)
    if not os.path.exists(local_path):
        print(f'Downloading {nc_file}...')
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print(f'{nc_file} already exists. Skipping download.')

