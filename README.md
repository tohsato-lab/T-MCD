# T-MCD

## Environment
- Ubuntu 20.04
- conda 4.10.3

## Installation
```shell
git clone https://github.com/tohsato-lab/T-MCD.git
cd T-MCD

# Download Dataset
wget https://github.com/tohsato-lab/T-MCD/releases/download/Latest/data.zip
unzip data.zip

# create conda environment
conda env create --file T-MCD.yaml
conda activate T-MCD
```

## Training and Testing
* dbScreen (40 second interval) -> WDDD
```shell
python3 main.py
```

If you want to validate other conditions, please change the options in **params.py**.

## About Dataset
### dbScreen
http://worm-srv1.mpi-cbg.de/dbScreen/movies/wildtype.mov 


The dbScreen dataset is a proprietary product of Cenix BioScience GmbH and the Max-Planck Gesellschaft zur Förderung der Wissenschaften e.V. and is protected under German Copyright Law and International Treaty Provisions.
* P. Gönczy, C. Echeverri, K. Oegema, A. Coulson, S. J. M. Jones, R. R. Copley, J. Duperon, J. Oegema, M. Brehm, E. Cassin, E. Hannak, M. Kirkham, S. Pichler, K. Flohrs, A. Goessen, S. Leidel, A.-M. Alleaume, C. Martin, N. Özlü, P. Bork, and A. A. Hyman, “Functional genomic analysis of cell division in C. elegans using RNAi of genes on chromosome III,” Nature, 408: 331–336, 2000. 


### WDDD
http://so.qbic.riken.jp/wddd/cdd/ 

<a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/3.0/88x31.png" /></a><br />The WDDD dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/">Creative Commons Attribution-ShareAlike 3.0 Unported License</a>.
* K. Kyoda, E. Adachi, E. Masuda, Y. Nagai, Y. Suzuki, T. Oguro, M. Urai, R. Arai, M. Furukawa, K. Shimada, J. Kuramochi, E. Nagai, and S. Onami, “WDDD: Worm Developmental Dynamics Database,” Nucleic Acids Research, 41: D732–D737, 2013.

In this study, we used wt_N2_030131_01 in the WDDD dataset.

### Annotation Data
<a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/3.0/88x31.png" /></a><br />The annotation data in the dbScreen and WDDD datasets are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/">Creative Commons Attribution-ShareAlike 3.0 Unported License</a>.

# Acknowledgements
The model of co-detection CNN included in this is borrowed from [co-detection CNN](https://github.com/naivete5656/WSCTBFP).
