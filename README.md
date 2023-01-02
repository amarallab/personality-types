# Personality Types

Code for data processing and clustering in the identification of personality types in:

M. Gerlach, B. Farb, W. Revelle, L.A.N. Amaral, A robust data-driven approach identifies four personality types across four large datasets, Nature Human Behaviour (2018).

**This repository is now archived and no changes will be made.**

## Data

As part of the repository we provide the position of 145,388 respondents in the 5-dimensional trait space of the Five-Factor-Model obtained from factor analysis of the responses to 300 items (Neuroticism, Extraversion, Openness, Agreeableness, Conscientiousness; see the paper for more details). This is the starting point for the identification of personality-types using our clustering approach running the ```analysis_clustering*``` notebooks.

This step can be reproduced from the original data in data.tar.gz (simply extract as data/) amd running the ```preprocessing_*``` notebooks. The data can be downloaded from here: https://osf.io/tbmh5/ 

We are not allowed to share the other datasets. The BBC-data is publicly available (upon registration): http://doi.org/10.5255/UKDA-SN-7656-1 . The mypersonality data is not available anymore, see here: http://mypersonality.org/

## Usage

#### Clustering

```analysis_clustering-01_number-of-clusters-BIC``` fits a Gaussian Mixture Model to the positions of the respondents in the 5D trait space varying the number of clusters. Using the Bayesian Information Criterion (BIC) we can determine the optimal number of cluster.

```analysis_clustering-02_meaningful-clusters-kernel-density``` determines whether a cluster solution is spurious or meaningful. Fixing the number of clusters (e.g. the optimal number determined via the BIC), for each cluster center we determine the density at its position in the 5D-space and compare with the expected density from a randomized dataset. This yields two quantities:
    - p-value: indicating whether the true density is significantly larger than the randomized density (low p-value == significant)
    - enrichment: the ratio between the true density and the average of the different realizations of the randomized densities 

#### Pre-processing

```preprocessing_01-filter-data``` reads Johnson's orginal data and puts it in a csv-file for easier usage, e.g. using pandas

```preprocessing_02-questions-vs-domains``` creates a file storing association between item and trait

```preprocessing_03-factor-analysis``` performs factor analysis and saves the factor scores as the positions of the 145,388 respondents in the 5-dimensional trait space. For this to work you have to specify the path to this folder in src/analysis/factor_analysis.py (```path_project = '/DRIVE/REPOS/personality-types-shared/'```) -- to be fixed

```preprocessing_04-demographics``` parses the csv file for extracting age and gender for follow-up analysis.


#### Plotting

```plotting_factor-loadings``` plots the factor loadings

```plotting_factor-scores-mrginals``` plots the 1D and 2D marginals of the factor scores

```plotting_gender-age``` plots gender and age distribution of the dataset

```plotting_gender-age-per-type``` plots the relative frequency of gender and age in each clusters.

## Setup

Clone the repository

```git clone https://github.com/amarallab/personality-types.git```

Install all required packages (here via conda):

- ```conda create -n personality-types python=3.6``` creates an environment
- ```source activate personality-types``` activates the environment
- ```conda install numpy scikit-learn matplotlib pandas openpyxl jupyter``` installs all the packages
- ```python -m ipykernel install --user --name personality-types --display-name "personality-types"``` makes the environemnt available in the jupyter notebooks
- ```jupyter notebook``` starts jupyter notebooks 

## Notes

#### External packages

For factor rotation we use the code from https://github.com/mvds314/factor_rotation which is in src/external/
