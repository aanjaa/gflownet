module load python/3.9 cuda/11.7 
cd venvs
virtualenv gflownet_update
source gflownet_update/bin/activate
cd ..
pip install ViennaRNA
pip install torch_cluster-1.6.1+pt113cu117-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.1+pt113cu117-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.17+pt113cu117-cp39-cp39-linux_x86_64.whl
cd ~/gflownet
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -e . --find-links https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install --upgrade torch_geometric
pip install wandb
pip install -U "ray[air]"
pip install PyTDC
pip install levenshtein
pip install flexs