# Prefix Tuning

The training and decoding scripts are in transformers/examples/*


1. Table-to-text training codes are in transformers/examples/control; the main training script is run_language_modeling.py. 

2. Table-to-text decoding codes are in transformers/examples/text-generation; the main script is text_generation.py. 

3. Summarization training & inference codes are in transformers/examples/seq2seq; the main script is finetuning.py


(Some of the file naming is not precise, will revise in later versions)

The two primary scripts I used to run my codes are `` train_e2e.py`` and ``train_bart.py``.

I use ``train_e2e.py`` (for table-to-text) and ``train_bart.py`` (for summarization) to submit my jobs to the SLURM queue; 
they are set to default of good hyperparameters, and can be used to tune hyperparameter :) Note that the path to datasets are specified in these two files.
To quickly setup and run the code: 

(1) 
``conda env create -f environment.yml``

(2)
``cd transformer; pip install -e .``

To train via prefix-tuning:

```python
cd transformers/examples/control; mkdir webnlg_models;

python train_e2e.py --optim_prefix yes --preseqlen 5 --epoch 5 --learning_rate 0.00005 --mode webnlg --bsz 5 --seed 101
```

To decode: 
```python
cd transformers/examples/text-generation;

python gen.py {data2text/webnlg/triples} yes yes {checkpoint_path} no
```

```python
python train_bart.py --mode xsum --preseqlen 200 --do_train yes --fp16 yes --bsz 16  --epoch 30  --gradient_accumulation_step 3 --learning_rate 0.00005  --mid_dim 800
```


Other baseline approaches 
```python
python train_e2e.py --tuning_mode {finetune/adaptertune} --epoch 5 --learning_rate 0.00005 --mode webnlg --bsz 5 --seed 101
```


For details of the methods and results, please refer to our paper. 

```bibtex
@misc{li2021prefixtuning,
      title={Prefix-Tuning: Optimizing Continuous Prompts for Generation}, 
      author={Xiang Lisa Li and Percy Liang},
      year={2021},
      eprint={2101.00190},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
