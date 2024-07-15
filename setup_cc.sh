module load StdEnv/2020 gcc/11.3.0
module load python/3.10 cuda/12.2 httpproxy
virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
# only necessary for RNA task
module load viennarna/2.5.1
# conda install -c bioconda viennarna -y

cd ~/wheels
# wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_cluster-1.6.1%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
# wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_scatter-2.1.1%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
# wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_sparse-0.6.17%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
pip install --no-index --find-links ~/wheels torch_cluster-1.6.3+pt21cu121-cp310-cp310-linux_x86_64.whl
pip install --no-index --find-links ~/wheels torch_scatter-2.1.2+pt21cu121-cp310-cp310-linux_x86_64.whl
pip install --no-index --find-links ~/wheels torch_sparse-0.6.18+pt21cu121-cp310-cp310-linux_x86_64.whl
cd ~/gflownet
pip install  --no-index --find-links ~/wheels torch==2.1.2+cu121 # --extra-index-url https://download.pytorch.org/whl/cu117
pip install -e . --no-index --find-links ~/wheels
pip install --no-index --find-links ~/wheels levenshtein wandb 
pip install --no-index --find-links ~/wheels rliable
pip install --no-index --find-links ~/wheels pyro-ppl==1.8.6
# only necessary for RNA task
# pip install flexs
pip install --no-index --find-links ~/wheels ray==2.9.2
