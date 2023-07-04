conda create -n $ENV_NAME python=3.9
conda activate $ENV_NAME

conda install openblas-devel -c anaconda
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install yapf plyfile pyyaml

export CUDA_HOME=/usr/local/cuda-11.1
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas