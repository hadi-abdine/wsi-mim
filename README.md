#  Word Sense Induction with Hierarchical Clustering and Mutual Information Maximization (MIM)
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

This repository contains the code to reporoduce the results of the paper "Word Sense Induction with Hierarchical Clustering and Mutual Information Maximization" by Hadi Abdine, Moussa Kamal Eddine, Michalis Vazirgiannis and Davide Buscaldi.


## Setup

Recommended environment is Python >= 3.8 and PyTorch 1.12, although earlier versions of Python and PyTorch may also work.

```bash
git clone https://github.com/hadi-abdine/wsi-mim.git
cd wsi-mim
bash ./data/download_fairseq.sh
bash ./data/download_SemEval.sh
conda create -n wsimim python=3.8
conda activate wsimim
pip install -r Requirements.txt
pip install -e ./fairseq/
```

Java is required to run SemEval 2010 and SemEval 2013 evaluation.
The training of the experiments is done using fp16 on NVIDIA Quadro RTX6000 24GB.

###### Optional:

1- Download the generated paraphrases during our experiments to skip the step of generating paraphrases:

```bash
cd WSI-MIM
bash ./data/download_paraphrases.sh
```


2- Download the obtained language model's word vectors of SemEval-2010 and SemEval-2013 datasets during our experiments to skip the step of getting the word vectors:

```bash
cd WSI-MIM
bash ./data/download_2010_vectors.sh
bash ./data/download_2013_vectors.sh
```


3- Download the obtained word sense vectors of SemEval-2010 and SemEval-2013 after the training of our MIM model during our experiments to skip the step of model training:

```bash
cd WSI-MIM
bash ./data/download_2010_results.sh
bash ./data/download_2013_results.sh
```

## Training and Evaluation

As described in the paper, the method has four main steps:




#### 1- Prepare the data by creating the randomly perturbated replicas using `transform_text.py`:

```bash
python ./code/transform_text.py --read_path ./data/SemEval-2013/contexts/txt-train/ \         
                                --write_path ./data/SemEval-2013/contexts/txt-train-masks4/ \
                                --percentage 0.4
```
                             
where:                             
* `--read_path`: is the directory that contains the original sentences of the target words. where each file in this path contains a list of sentences of a target word and each line is a training example.
* `--write_path`: is the directory to write the paraphrases file of each target word. 
* `--percentage`: the percentage of words to be masked in the training example.
* for SemEval-2010, read_path is: `./data/SemEval-2010/train/all/`  for the training set. And `./data/SemEval-2010/test/all/` for the test set.
* for SemEval-2013, read_path is: `./data/SemEval-2013/contexts/txt-train/` for the training set. And `./data/SemEval-2013/contexts/txt-test/` for the test set.

this step should be done for both training and testing set, it takes around 1.5 hours on NVIDIA Quadro RTX6000 GPU.                      
<b>The paraphrases could be downloaded (see optional section above).</b>




    
#### 2- Get the contextual word embeddings using `get_vectors.py`:

```bash
python ./code/get_vectors.py --path ./data/SemEval-2013/contexts/txt-train/ \
                             --model roberta \
                             --split train \
                             --save ./data/SemEval-2013/contexts/vectors/ \
                             --layer 17 
```

where:
* `--path`: is the directory that contains the training sentences of each target word.
* `--save`: is the directory to save vectors to.
* `--model`: is the pretrained model to use, you can choose between roberta, bert, deberta or bart.
* `--split`: is to specify whether you are working with thr training or testing split (used to name the generated file), choose between train or test.
* `--layer`: is to specify the hidden layer from which to extract the vectors.
    
This command that has the highest cost in our model, it takes around 1.5 hours per run and should be done for `train_set`, `train_paraphrase_set`, `test_set` and `test_paraphrases_set`.
<b>The word vectors could be downloaded (see optional section above).</b>
    
    
    
    
#### 3- Train the MIM model using `train_iic.py`:

```bash
python ./code/train_iic.py --train_path ./data/SemEval-2013/contexts/vectors/train_vectors_roberta17_.txt \
                           --test_path ./data/SemEval-2013/contexts/vectors/test_vectors_roberta17_.txt \
                           --train_mask_path ./data/SemEval-2013/contexts/vectors/train_vectors_roberta17_masks4.txt \
                           --test_mask_path ./data/SemEval-2013/contexts/vectors/test_vectors_roberta17_masks4.txt \
                           --text_path ./data/SemEval-2013/contexts/txt-train \
                           --batch_size 64 \
                           --training_epochs 5 \
                           --dataset semeval-2013 \
                           --eval_path ./data/SemEval-2013/scoring/ \
                           --gold_path ./data/SemEval-2013/keys/gold/all.key \
                           --ids_list_path ./data/SemEval-2013/contexts/ids/ \
                           --output_vectors_path ./data/SemEval-2013/results/iic_roberta17_7r_masks4_vectors_bestts12_tmp.key \
                           --nb_clusters 7 \
                           --dynamic 0 \ 
                           --output_vectors_path ./data/SemEval-2013/results/iic_roberta17_7r_masks4_vectors_bestts12_tmp.key
```

where:                        
* `--train_path`: path to the training vectors generated in step 2.
* `--test_path`: path to the test word vectors generated in step 2. 
* `--train_mask_path`: path to the train paraphrases word vectors generated in step 2. 
* `--test_mask_path`: path to the test paraphrases word vectors generated in step 2. 
* `--text_path`: path to the original sentences. (`./data/SemEval-2013/contexts/txt-train/` for SemEval-2013 or `./data/SemEval-2013/contexts/txt-train/` for SemEval-2010).
* `--dataset`: semeval-2013 or semeval-2010
* `--eval_path`: path of the java evaluation file:
  - for SemEval-2010: `./data/SemEval-2010/evaluation/unsup_eval/`
  - for SemEval-2013: `./data/SemEval-2013/scoring/`
* `--gold_path`: path of the golden annotations, 
  - for SemEval-2010: `./data/SemEval-2010/evaluation/unsup_eval/keys/all.key`
  - for SemEval-2013: `./data/SemEval-2013/keys/gold/all.key`
* `--ids_list_path` : path of the ids file of semeval-2013 test set,
  - for SemEval-2010: `None`
  - for SemEval-2013: `./data/SemEval-2013/contexts/ids/`
* `--output_vectors_path` : path to save the sense embeddings.
* `--nb_clusters`: number of fixed clusters. 
* `--dynamic`: `1` for dynamic number of clusters and `0` for fixed number of clusters.
                                        
                       
<b>The sense embeddings could be downloaded (see optional section above).</b>                     
                       
                       
#### 4- Evaluation: using `evaluate.py`:

```shell
python ./code/evaluate.py --vectors_path ./data/SemEval-2013/results/iic_roberta17_7r_masks4_vectors_bestts12_tmp.key \
                          --dataset semeval-2013 \
                          --nb_clusters 7 \
                          --dynamic 0 \
                          --ids_list_path ./data/SemEval-2013/contexts/ids/ \
                          --eval_path ./data/SemEval-2013/scoring/ \
                          --gold_path ./data/SemEval-2013/keys/gold/all.key \
                          --files_path ./data/SemEval-2013/contexts/txt-test/
```                   
                       
where:                        
* `--vectors_path`: path to the sense embeddings generated in step 3 (output_vectors_path in step 3). 
* `--dataset`: semeval-2013 or semeval-2010
* `--eval_path`: path of the java evaluation file:
  - for SemEval-2010: `./data/SemEval-2010/evaluation/unsup_eval/`
  - for SemEval-2013: `./data/SemEval-2013/scoring/`
* `--gold_path`: path of the golden annotations, 
  - for SemEval-2010: `./data/SemEval-2010/evaluation/unsup_eval/keys/all.key`
  - for SemEval-2013: `./data/SemEval-2013/keys/gold/all.key`
* `--ids_list_path`: path of the ids file of semeval-2013 test set,
  - for SemEval-2010: `None`
  - for SemEval-2013: `./data/SemEval-2013/contexts/ids/`
* `--nb_clusters`: number of fixed clusters. 
* `--dynamic`: `1` for dynamic number of clusters and `0` for fixed number of clusters.
* `--files_path`: path to the original sentences. (`./data/SemEval-2013/contexts/txt-train/` for SemEval-2013 or `./data/SemEval-2013/contexts/txt-train/` for SemEval-2010).

This repository is tested and verified on CPU, NVIDIA cuda acceleration, and Pytorch accelaration for Apple silicon.
