# AndreiN, 2019

import glob
import os
from os import listdir

# ======================================================================================================================
# Clean failed experiments

except_dir = []
except_dir = [os.path.dirname(x) for x in except_dir]

experiment_path = "results/2019Feb14-164238_openai_rnd"

cfg_files = glob.glob(f"{experiment_path}/**/cfg.yaml", recursive=True)

selected = []
for run_index, cfg_file in enumerate(cfg_files):
    dir_name = os.path.dirname(cfg_file)

    if dir_name not in except_dir:
        selected.append(dir_name)

filter_files = [".__leaf", "cfg.yaml"]
for dir_name in selected:
    all_files = [f"{dir_name}/{x}" for x in listdir(f"{dir_name}/")]
    fs = [x for x in all_files if not any([y in x for y in filter_files])]
    for f in fs:
        os.remove(f)
# ======================================================================================================================

import torch

z = torch.rand(4, 10, requires_grad=True)
t = torch.rand(4, 10, requires_grad=True)
ln = torch.nn.Linear(10, 10)

out = z + t
out = ln(out)
out = out.sum()


adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(ln.parameters()),
            allow_unused=True, retain_graph=True
        )