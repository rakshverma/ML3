import json
import os

config_path = 'configs/open_candidate_run.json'
with open(config_path, 'r') as f:
    config = json.load(f)

base_dir = '/Users/rakshitverma/Documents/ML3'
# New values
before_id = "S2A_43PDT_20240124_0_L2A"
before_date = "2024-01-24"
after_id = "S2B_43PDT_20250113_0_L2A"
after_date = "2025-01-13"
origin_x = 74.48
origin_y = 15.83
pixel_width = 0.0001
pixel_height = 0.0001

config['before_scene'].update({
    "acquired_on": before_date,
    "source": f"Sentinel-2 L2A {before_id}",
    "origin_x": origin_x,
    "origin_y": origin_y,
    "pixel_width": pixel_width,
    "pixel_height": pixel_height,
    "npz_path": os.path.join(base_dir, "data/open/site_001/before_scene.npz")
})

config['after_scene'].update({
    "acquired_on": after_date,
    "source": f"Sentinel-2 L2A {after_id}",
    "origin_x": origin_x,
    "origin_y": origin_y,
    "pixel_width": pixel_width,
    "pixel_height": pixel_height,
    "npz_path": os.path.join(base_dir, "data/open/site_001/after_scene.npz")
})

config['boundary']['path'] = os.path.join(base_dir, "outputs/open_udyambag/industrial_candidates.geojson")
config['output_dir'] = os.path.join(base_dir, "outputs/open_candidate_run")

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("Config updated with new transform and absolute paths")
