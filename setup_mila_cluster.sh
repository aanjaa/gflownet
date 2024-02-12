#!/bin/bash

module load miniconda/3 cuda/11.7
conda create -p $SLURM_TMPDIR/env python=3.9 -y
conda activate $SLURM_TMPDIR/env
conda install -c bioconda viennarna -y

cd $SLURM_TMPDIR
wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_cluster-1.6.1%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_scatter-2.1.1%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_sparse-0.6.17%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
pip install torch_cluster-1.6.1+pt113cu117-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.1+pt113cu117-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.17+pt113cu117-cp39-cp39-linux_x86_64.whl
cd ~/repos/eval_gfn_repo
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -e . --find-links https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install PyTDC flexs levenshtein wandb
pip install ray==2.6.2
