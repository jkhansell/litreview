#!/bin/bash
#SBATCH --job-name=PaperAnalysis
#SBATCH --partition=nukwa
#SBATCH --nodes=1 
#SBATCH --nodelist=nukwa-05.cnca
#SBATCH --time=4:00:00
#SBATCH --output=paper_analysis_%j.out
#SBATCH --mail-user=jovillalobos@cenat.ac.cr
#SBATCH --mail-type=END,FAIL

source /home/jovillalobos/.bashrc

conda activate pytorch-cuda
python paper_analysis.py