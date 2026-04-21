# Shrub Detection Pipeline - Submission Runner

This directory contains a reproducible Jupyter notebook that executes the complete shrub detection ML pipeline end-to-end: **Modeling → Postprocessing → Evaluation**.

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create --name shrub_env python
conda activate shrub_env

# Navigate to this directory
cd shrub

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook main.ipynb
```

Crucial:
The data is expensive to fetch and generate so you can execute the sections in order, or copy over provided gdrive credentials which are onyl valid for this remote artifacts folder linked here: https://drive.google.com/drive/folders/1F9AtiUfx_z48tQIkNbDpzZUpH6R5uoCN?usp=drive_link

Execute sections in order but to just see it all in action you can skip straight to the final large modeling/postprocessing section provided the data exists locally.

## Flowchart
For a high level overview of the components contained in `/Final` see our [Lucid Chart](https://lucid.app/lucidchart/32a2e4fe-7cf1-4b10-921b-789723bfd876/edit?invitationId=inv_f4a3a702-681c-4030-95ef-ffcd3ce97884&page=0_0)
