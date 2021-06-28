1.
Create a virtualenv (python 3.6) in project root directory.
sudo apt-get install python3.6-dev python3.6-venv
python3.6 -m venv myenv

2.
Open a terminal in your project root directory.

3.
Activate virtualenv.
source myenv/bin/activate


4. 
python -m pip install --upgrade setuptools
python -m pip install --upgrade pip

5.
pip install -r requirements.txt 

APEX:
git clone https://github.com/NVIDIA/apex
cd apex

before install (in terminal in apex dir):
export CUDA_HOME=/usr/local/cuda

intalling gcc 8 needed?

Comment out part (torch 1.2) - newer versions might not need this:

"""
ext_modules.append(
    CUDAExtension(name='mlp_cuda',
                  sources=['csrc/mlp.cpp',
                           'csrc/mlp_cuda.cu'],
                  extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                      'nvcc':['-O3'] + version_dependent_macros}))
"""

Install command:
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
