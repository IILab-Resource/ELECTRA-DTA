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
 
### download dataset

```
cd ELECTRA-DTA
wget -np -nH --cut-dirs 2 -r -A .csv,.pkl http://iilab.hit.edu.cn/dtadata/ElectraDTA
```
