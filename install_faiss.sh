#!/bin/bash

# install conda
wget -c https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh # replace by required version
chmod +x Anaconda3-5.1.0-Linux-x86_64.sh
bash ./Anaconda3-5.1.0-Linux-x86_64.sh -b -f -p /usr/local
conda install -y --prefix /usr/local -c /root

# install faiss for gpu - ensures fast computation
conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA10

# torch may get uninstalled during the previous process; needs pip preinstalled
pip install torch
