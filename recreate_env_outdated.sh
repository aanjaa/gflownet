deactivate
cd ..
module load python/3.9 cuda/11.7 
cd venvs
virtualenv gflownet 
source gflownet/bin/activate
cd ..
cd gflownet
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -e . --find-links https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install --upgrade torch_geometric
pip install wandb
pip install -U "ray[air]"
pip install PyTDC
