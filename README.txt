# INSTALL PRE-REQS
sudo yum install git


# INSTALL CONDA
wget –P . https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
sudo chmod +x Anaconda3-2023.03-1-Linux-x86_64.sh 
./Anaconda3-2023.03-1-Linux-x86_64.sh
# exit and re-enter the shell after installing Anaconda
# (base) context should appear in Linux prompt


# CONFIGURE ENVIRONMENT
conda create --name llama python=3.8.16
conda activate llama
python --version
conda install --name llama --file conda-spec.txt
conda install pip
pip install -r requirements.txt


# DOWNLOAD PRE-TRAINED BASE MODEL
cd scripts
python download.py --repo_id openlm-research/open_llama_3b --local_dir  ../models

# FINE-TUNE MODEL
python finetune-basic.py

# RUN INFERENCE
python infer-basic.py
