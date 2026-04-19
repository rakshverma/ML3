import json
import numpy as np
import geopandas as gpd
from rasterio import features
from affine import Affine

def run():
    config_path = "configs/open_candidate_run.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load before_scene
    before_scene_path = config["before_scene"]["npz_path"]
    before_data = np.load(before_scene_path)
    
    red = before_data['red'].astype(float)
    nir = before_data['nir'].astype(float)
    # Sentinel-2 data is often in DN (Digital Numbers), usually 0-10000
    # But NDVI ratio stays the same
    ndvi = (nir - red) / (nir + red + 1e-8)

    def get_stats(feat_idx, gdf_path):
        gdf = gpd.read_file(gdf_path)
        geom = gdf.geometry.iloc[feat_idx]
        
        # Transform from config
        transform = Affine.from_gdal(
            config["before_scene"]["origin_x"],
            config["before_scene"]["pixel_width"],
            0,
            config["before_scene"]["origin_y"],
            0,
            config["before_scene"]["pixel_height"]
        )
        
        mask = features.rasterize(
            [(geom, 1)],
            out_shape=ndvi.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
        
        mask_pixels = int(np.sum(mask))
        
        # Bbox pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if any(rows):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox_pixels = int((rmax - rmin + 1) * (cmax - cmin + 1))
        else:
            bbox_pixels = 0
        
        ndvi_full_mean = float(np.mean(ndvi))
        ndvi_full_max = float(np.max(ndvi))
        
        if mask_pixels > 0:
            ndvi_mask = ndvi[mask == 1]
            ndvi_mask_mean = float(np.mean(ndvi_mask))
            ndvi_mask_max = float(np.max(ndvi_mask))
            count_gt_035 = int(np.sum(ndvi_mask > 0.35))
        else:
            ndvi_mask_mean = 0.0
            ndvi_mask_max = 0.0
            count_gt_035 = 0
            
        return {
            "idx": feat_idx,
            "mask_count": mask_pixels,
            "bbox_pixels": bbox_pixels,
            "full_mean": ndvi_full_mean,
            "full_max": ndvi_full_max,
            "mask_mean": ndvi_mask_mean,
            "mask_max": ndvi_mask_max,
            "count_gt_035": count_gt_035
        }

    res7 = get_stats(config["boundary"]["feature_index"], config["boundary"]["path"])
    
    def print_res(r):
        print(f"Index {r['idx']}:")
        print(f"Mask Pixels: {r['mask_count']}")
        print(f"Bbox Pixels: {r['bbox_pixels']}")
        print(f"Full NDVI Mean/Max: {r['full_mean']:.4f} / {r['full_max']:.4f}")
        print(f"Mask NDVI Mean/Max: {r['mask_mean']:.4f} / {r['mask_max']:.4f}")
        print(f"NDVI > 0.35: {r['count_gt_035']}")

    print_res(res7)
    
    if res7["mask_count"] < 200:
        print("\nComparing with Indices 0 and 1:")
        print_res(get_stats(0, config["boundary"]["path"]))
        print_res(get_stats(1, config["boundary"]["path"]))

run()
