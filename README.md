# contrast_LM

The training and evaluation scripts are in transformer/examples/*
Currently the naming is not precise, will revise: 
all the table-to-text training codes are in transformer/examples/control; the main training script is run_language_modeling.py. 
all the table-to-text inference codes are in transformer/examples/text-generation; the main script is text_generation.py. 
all the summarization training & inference codes are in transformer/examples/seq2seq; the main script is finetuning.py

I use train_e2e.py (for table-to-text) and train_bart.py (for summarization) to submit my jobs to the queue; they are set to default of good hyperparameters, and we can be used to tune hyperparameter :) 

To quickly setup and run the code: 
(1) 
```python
cd transformer; pip install -e .
```

```python
python train_e2e.py --optim_prefix yes --preseqlen 5 --epoch 5 --learning_rate 0.00005 --mode webnlg --bsz 5 --seed 101
```
Other baseline approaches 
```python
python train_bart.py --tuning_mode {finetune/adaptertune} --epoch 5 --learning_rate 0.00005 --mode webnlg --bsz 5 --seed 101
```
