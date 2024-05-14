FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# # Set up the Conda environment
# COPY requirements.yml requirements.yml
# RUN conda env create -f requirements.yml

# # Additional setup steps