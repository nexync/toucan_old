# Toucan: Token-Aware Character Level Language Modeling

![grab-landing-page](https://github.com/wfleshman/toucan/blob/main/media/toucan.gif)

[**Repository**](#repository) | [**Toucan**](#toucan) | [**Data**](#data) | [**Training**](#training) | [**Inference**](#inference) | [**Cite**](#cite)

Paper: [Toucan](https://arxiv.org/abs/2311.08620)

## Repository:

Repository is a fork from: https://github.com/PiotrNawrot/dynamic-pooling.

Our project builds directly from their work and heavily uses their codebase. To make comparisons easy we've marked additions and changes to their code with comments of the form:
```#<CHANGE> ... #</CHANGE> ... #<ADDITION> ... #</ADDITION>```

Additions/Changes:
- ./configs/ 
    - we've added configs for training the toucan (x5) and toucan (x10) models.
- hourglass.py
    - several modifications to "toucanize" the model.
- train.py
    - minor changes to make the code compatible with the new hourglass.py.
- generate.py
    - new file to demonstrate tokenization and generation.
- ./scripts/run_inf.sh
    - script for invoking generate.py
 
## Toucan:

Toucan is a technique for adding token-awareness to character-level language models which include a token-like pooling or patch selection mechanism. It works by inserting special end-of-token vectors into the sequence at the locations designated by the original architecture. It then adjusts the labels so that the last character in each token predicts an end-of-token, and the end-of-token predicts the first character in the next token. 

After training, a toucanized decoder can generate characters for the entire token without reprocessing the sequence with the rest of the model. Here's an animation of the toucanized hourglass transformer used in this work:

![grab-landing-page](https://github.com/wfleshman/toucan/blob/main/media/model.gif)

## Data:
- Same download & preprocess steps as original repository
    - text8
        - `bash scripts/get_text8.sh` 
    - wiki40b 
        - `bash scripts/get_wiki40b.sh $lang`
        - where $lang is for example `vi`
        - check [Link](https://www.tensorflow.org/datasets/catalog/wiki40b) for how the abbreviation of other languages
        - Script first downloads wiki40b under `./data/wiki40b/$lang/`, and then applies our cleaners on top of it based on [text8](http://mattmahoney.net/dc/textdata) cleaning rules. Final training data sits under `./data/wiki40b/$lang/text8`. We found that for some systems there might occur some errors when downloading wiki40b using `datasets`. In this case after you manage to get the data just apply our cleaners on it.
- Train Unigram
    - `python tokenizer_data/train_tokenizer.py $vocab_size $dataset`
    - `$vocab_size` is the integer target vocab size of Unigram
    - `$dataset` is `text8` for text8, `wiki40b/$lang/text8` for wiki40b

## Training:

- To run training use:
```
C=configs/toucan10.yaml
GPUS=N
bash scripts/run_exp.sh
```
    - C -> defines the path to the config 
    - GPUS -> defines the number of GPUs for distributed run, when not given then the training runs on a single GPU/CPU

## Inference:
- To run inference use:
```
C=configs/toucan10.yaml
GPUS=N
bash scripts/run_inf.sh
```

## Cite:

```
@misc{fleshman2023toucan,
      title={Toucan: Token-Aware Character Level Language Modeling}, 
      author={William Fleshman and Benjamin Van Durme},
      year={2023},
      eprint={2311.08620},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
