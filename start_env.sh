pip install -r requirements.txt
conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -y tensorflow-gpu
git clone https://github.com/maximecb/gym-minigrid.git
cd gym-minigrid
pip install -e .
cd ../
# git clone https://github.com/lcswillems/torch-rl.git
# cd torch-rl
# pip install -e torch_rl
cd ../
git clone https://github.com/tudor-berariu/liftoff.git
cd liftoff/
pip install -e . --process-dependency-links
cd ../