pip install -r requirements.txt
pip install -e torch_rl
conda install -y pytorch torchvision cudatoolkit=9.2 -c pytorch
conda install -y tensorflow-gpu
cd ../
git clone https://github.com/maximecb/gym-minigrid.git
cd gym-minigrid
pip install -e .
cd ../
# git clone https://github.com/lcswillems/torch-rl.git
# cd torch-rl
# cd ../
git clone https://github.com/tudor-berariu/liftoff.git
cd liftoff/
pip install -e . --process-dependency-links
cd ../