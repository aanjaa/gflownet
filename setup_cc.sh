module load python/3.9 cuda/11.7 httpproxy
virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
# only necessary for RNA task
# conda install -c bioconda viennarna -y

cd ~/wheels
# wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_cluster-1.6.1%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
# wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_scatter-2.1.1%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
# wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_sparse-0.6.17%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
pip install --no-index --find-links ~/wheels torch_cluster-1.6.1+pt113cu117-cp39-cp39-linux_x86_64.whl
pip install --no-index --find-links ~/wheels torch_scatter-2.1.1+pt113cu117-cp39-cp39-linux_x86_64.whl
pip install --no-index --find-links ~/wheels torch_sparse-0.6.17+pt113cu117-cp39-cp39-linux_x86_64.whl
cd ~/gflownet
pip install  --no-index --find-links ~/wheels torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -e . --no-index --find-links ~/wheels
pip install --no-index --find-links ~/wheels levenshtein wandb 
pip install --no-index --find-links ~/wheels rliable
pip install --no-index --find-links ~/wheels pyro-ppl==1.8.6
# only necessary for RNA task
# pip install flexs
pip install --no-index --find-links ~/wheels ray==2.6.3
