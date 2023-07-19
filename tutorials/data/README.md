## Download example data 
Download the preprocessed data from the NeurIPS 2021 competition "Multimodal Single-Cell Data Integration" from GEO (GEO accession: GSE194122; Supplementary file: comprises preprocessed data).

```
# Download data from GEO 
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122%5Fopenproblems%5Fneurips2021%5Fmultiome%5FBMMC%5Fprocessed%2Eh5ad%2Egz

# Unzip file
zcat GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz > GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad

```