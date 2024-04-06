module load miniconda/3 cudatoolkit/12.2
conda create -p $SLURM_TMPDIR/env python=3.10 -y
conda activate $SLURM_TMPDIR/env
# only necessary for RNA task
# conda install -c bioconda viennarna -y

cd $SLURM_TMPDIR
wget https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_cluster-1.6.3%2Bpt21cu121-cp310-cp310-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_scatter-2.1.2%2Bpt21cu121-cp310-cp310-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_sparse-0.6.18%2Bpt21cu121-cp310-cp310-linux_x86_64.whl
pip install torch_cluster-1.6.3+pt21cu121-cp310-cp310-linux_x86_64.whl
pip install torch_scatter-2.1.2+pt21cu121-cp310-cp310-linux_x86_64.whl
pip install torch_sparse-0.6.18+pt21cu121-cp310-cp310-linux_x86_64.whl
cd ~/gflownet
pip install torch==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install PyTDC levenshtein wandb 
pip install -U rliable
# only necessary for RNA task
# pip install flexs
pip install ray
