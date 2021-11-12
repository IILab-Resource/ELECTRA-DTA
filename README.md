# ELECTRA-DTA

In this github repository, you will see all the source code. However, the
trained data is too big (total 85G). The entire trained data is in the dataset
subfolder of the following url.

http://iilab.hit.edu.cn/dtadata/ElectraDTA/ 

## dependency

### conda environment

```
conda create -n ElectraDTA python=3.6
conda activate ElectraDTA
```
### packages

```
pip install tensorflow-gpu==1.14
pip install keras==2.2.5
pip install rlscore sklearn tqdm
``` 

### clone

```
git clone https://github.com/IILab-Resource/ELECTRA-DTA
```
 
### Run scripts
python DTA-BindingDB-Ki.py: this script use the refined BindingDB dataset with average 12 layers electra embedding.
python DTA-BindingDB-Full-average.py: this script use the original BindingDB dataset with average 12 layers electra embedding.
python DTA-KIBA-Full.py :  this script use the original BindingDB dataset with average 12 layers electra embedding.

for other dataset and embeddings, please change the dataset path in the python script files.  
Change the dataset:
``` 
data_file = 'dataset/BindingDB-full-data.csv'
```
to 
``` 
data_file = 'dataset/davis-full-data.csv'
```
change the embedding:
```
protein_seqs_emb  = load_dict('dataset/embedding256-12layers/atomwise_kiba-full_protein_maxlen1022_dim256-layer{}.pkl'.format(embedding_no))
smiles_seqs_emb = load_dict('dataset/embedding256-12layers/atomwise_kiba-full_smiles_maxlen100_dim256-layer{}.pkl'.format(embedding_no))

```
to 
```
protein_seqs_emb  = load_dict('dataset/embedding256-12layers/atomwise_davis-full_protein_maxlen1022_dim256-layer{}.pkl'.format(embedding_no))
smiles_seqs_emb = load_dict('dataset/embedding256-12layers/atomwise_davis-full_smiles_maxlen100_dim256-layer{}.pkl'.format(embedding_no))

```


### download dataset

```
cd ELECTRA-DTA
wget -np -nH --cut-dirs 2 -r -A .csv,.pkl http://iilab.hit.edu.cn/dtadata/ElectraDTA
```
