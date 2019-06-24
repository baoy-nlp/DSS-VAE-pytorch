## Code base: (https://github.com/baoy-nlp/TextVAE-pytorch)

## Requirements
- [ZPar](https://sourceforge.net/projects/zpar/files/0.7.5/zpar-0.7.5.tar.gz/download)(parsing for preprocess)
- PyTorch 0.4 +
- nltk
- tensorboardX
- Numpy
- PyYAML
- pickle (for dataset or model store and load)

## Data Preprocess
- tokenize

    `python preprocess/tokenize.py --raw_file [raw_file_path] --token_file [token_out_path] --for_parse`
- parse the data with ZPar
    
    `【ref to zpar】`
- prepare dataset
    - convert <Constituency Tree> to <Sentence, Linearized Tree>
    
        `python preprocess/tree_convert --tree_file [tree_file_path] --out_file [tree_out_path] --mode s2b`

    - generate dataset and vocabulary
        
        `python struct_self/generate_dataset.py --train_file [<Sentence,LinearTree> file] --dev_file [<Sentence,LinearTree> file] --test_file [<Sentence,LinearTree> file] --tgt_dir [output_dir] --max_src_vocab 30000 --max_src_len 30 --max_tgt_len 90 --train_size 100000`

- After Pre-Process, the prepared data directory structure is as follows:
    > SNLI-SVAE [Tgt Dir]
     > - train.bin
     > - test.bin
     > - dev.bin
     > - vocab.bin

## Training
- training from scratch with following command:

  `python main.py --config_files [config.yaml file] --mode train_vae --exp_name [exp_name:for note]`

