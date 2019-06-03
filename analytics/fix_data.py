import glob
import os
import re


# Fix cfgs - add cfg_id
def add_cfg_id_back(search_path):
    cfgs = glob.glob(f"{search_path}/**/cfg.yaml", recursive=True)
    for cfg_path in cfgs:
        exp_folder = os.path.basename(os.path.dirname(os.path.dirname(cfg_path)))
        match_cfg_id = int(re.match("^(\d+.)_", exp_folder).group(1))

        with open(cfg_path, "a") as myfile:
            myfile.write(f"\ncfg_id: {match_cfg_id}\n")

