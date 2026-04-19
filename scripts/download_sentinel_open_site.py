import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pystac_client import Client
import requests

def download_and_process():
    # Boundary: (74.483682, 15.8200644, 74.4845073, 15.8208624)
    # Give it some buffer
    bbox = [74.48, 15.81, 74.49, 15.83] 
    search_period_before = "2024-01-01/2024-01-31"
    search_period_after = "2025-01-01/2025-01-31"
    catalog_url = "https://earth-search.aws.element84.com/v1"
    
    client = Client.open(catalog_url)
    
    def get_best_item(period):
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=period,
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}]
        )
        items = list(search.items())
        if not items:
            raise Exception(f"No items found for period {period}")
        return items[0]

    item_before = get_best_item(search_period_before)
    item_after = get_best_item(search_period_after)

    asset_map = {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "nir": "nir",
        "swir": "swir16"
    }

    dst_crs = 'EPSG:4326'
    res = 0.0001
    
    # Use accurate bbox from before to define the output grid
    # To avoid offset issues, let's use the bbox requested
    dst_width = int((bbox[2] - bbox[0]) / res)
    dst_height = int((bbox[3] - bbox[1]) / res)
    dst_transform = rasterio.transform.from_origin(bbox[0], bbox[3], res, res)

    def process_item(item, filename):
        data_dict = {}
        scl_available = "scl" in item.assets
        
        for key, asset_name in asset_map.items():
            asset_url = item.assets[asset_name].href
            with rasterio.open(asset_url) as src:
                dst_array = np.zeros((dst_height, dst_width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )
                dst_array = np.clip(dst_array / 10000.0, 0, 1)
                data_dict[key] = dst_array

        if scl_available:
            scl_url = item.assets["scl"].href
            with rasterio.open(scl_url) as src:
                scl_raw = np.zeros((dst_height, dst_width), dtype=np.uint8)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=scl_raw,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
                cloud_mask = np.isin(scl_raw, [3, 8, 9, 10, 11])
        else:
            cloud_mask = np.zeros((dst_height, dst_width), dtype=bool)
            
        data_dict["cloud_mask"] = cloud_mask
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, **data_dict)
        return item.id, item.properties["datetime"], data_dict["blue"].shape

    id_b, date_b, shape_b = process_item(item_before, "data/open/site_001/before_scene.npz")
    id_a, date_a, shape_a = process_item(item_after, "data/open/site_001/after_scene.npz")

    print(f"BEFORE_ID: {id_b}")
    print(f"BEFORE_DATE: {date_b}")
    print(f"AFTER_ID: {id_a}")
    print(f"AFTER_DATE: {date_a}")
    print(f"SHAPE: {shape_a}")
    print(f"TRANSFORM: {bbox[0]}, {bbox[3]}, {res}, {res}")

download_and_process()
