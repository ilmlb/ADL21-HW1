# Homework 1 ADL NTU 110 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent classification and slot tagging datasets
bash preprocess.sh
```

## Intent classification
### Training
```shell
python train_intent.py [--data_dir dir] [--cache_dir dir] [--ckpt_dir dir] <--ckpt_name ckpt> [--max_len len] [--recurrent_struc rnn] [--hidden_size hidden] [--num_layers layer] [--dropout drop] [--bidirectional bool] [--lr lr] [--batch_size batch] [--device cuda] [--num_epoch epoch]
```
- `--data_dir`: Directory to the dataset. Default="./data/intent/"
- `--cache_dir`: Directory to the preprocessed caches. Default="./cache/intent/"
- `--ckpt_dir`: Directory to save the model file. Default="./ckpt/intent/"
- `--ckpt_name`: Name of model checkpoint. The path of the stored weight will be under `ckpt_dir/ckpt_name`. **This argument has no default value and must be given.**
- `--max_len`: Default=128
- `--recurrent_struc`: rnn|lstm|gru. Default="lstm"
- `--hidden_size`: Default=512
- `--num_layers`: Default=2
- `--dropout`: Default=0.1
- `--bidirectional`: Default=True
- `--lr`: Learning rate. Default=0.001
- `--batch_size`: Default=128
- `--device`: cpu|cuda. Default="cuda"
- `--num_epoch`: Default=100
### Testing
```shell
python test_intent.py <--test_file file> [--cache_dir dir] <--ckpt_path ckpt> [--pred_file file] [--max_len len] [--recurrent_struc rnn] [--hidden_size hidden] [--num_layers layer] [--dropout drop] [--bidirectional bool] [--batch_size batch] [--device cuda]
```
- `--test_file`: Path to the test file. 
- `--cache_dir`: Directory to the preprocessed caches. Default="./cache/intent/"
- `--ckpt_path`: Path to model checkpoint.
- `--pred_file`: Path to prediction file. Default="pred.intent.csv"
- `--max_len`: Default=128
- `--recurrent_struc`: rnn|lstm|gru. Default="lstm"
- `--hidden_size`: Default=512
- `--num_layers`: Default=2
- `--dropout`: Default=0.1
- `--bidirectional`: Default=True
- `--batch_size`: Default=128
- `--device`: cpu|cuda. Default="cuda"


## Slot tagging
### Training
```shell
python train_slot.py [--data_dir dir] [--cache_dir dir] [--ckpt_dir dir] <--ckpt_name ckpt> [--max_len len] [--recurrent_struc rnn] [--hidden_size hidden] [--num_layers layer] [--dropout drop] [--bidirectional bool] [--out_channels channel] [--kernel_size kernel] [--lr lr] [--batch_size batch] [--device cuda] [--num_epoch epoch] [--loss loss]
```
- `--data_dir`: Directory to the dataset. Default="./data/slot/"
- `--cache_dir`: Directory to the preprocessed caches. Default="./cache/slot/"
- `--ckpt_dir`: Directory to save the model file. Default="./ckpt/slot/"
- `--ckpt_name`: Name of model checkpoint. The path of the stored weight will be under `ckpt_dir/ckpt_name`. **This argument has no default value and must be given.**
- `--max_len`: Default=128
- `--recurrent_struc`: rnn|lstm|gru. Default="lstm"
- `--hidden_size`: Default=512
- `--num_layers`: Default=2
- `--dropout`: Default=0.1
- `--bidirectional`: Default=True
- `--out_channels`: The number of output channels of convolution layers of CNN-LSTM. Default=100
- `--kernel_size`: The kernel size of convolution layers of CNN-LSTM must be odd. Default=3
- `--lr`: Learning rate. Default=0.001
- `--batch_size`: Default=128
- `--device`: cpu|cuda. Default="cuda"
- `--num_epoch`: Default=100
- `--loss`: Loss function. ce|focal. Default="ce"
### Testing
```shell
python test_slot.py <--test_file file> [--cache_dir dir] <--ckpt_path ckpt> [--pred_file file] [--max_len len] [--recurrent_struc rnn] [--hidden_size hidden] [--num_layers layer] [--dropout drop] [--bidirectional bool] [--out_channels channel] [--kernel_size kernel] [--batch_size batch] [--device cuda]
```
- `--test_file`: Path to the test file. 
- `--cache_dir`: Directory to the preprocessed caches. Default="./cache/slot/"
- `--ckpt_path`: Path to model checkpoint.
- `--pred_file`: Path to prediction file. Default="pred.slot.csv"
- `--max_len`: Default=128
- `--recurrent_struc`: rnn|lstm|gru. Default="lstm"
- `--hidden_size`: Default=512
- `--num_layers`: Default=2
- `--dropout`: Default=0.1
- `--bidirectional`: Default=True
- `--out_channels`: The number of output channels of convolution layers of CNN-LSTM. Default=100
- `--kernel_size`: The kernel size of convolution layers of CNN-LSTM must be odd. Default=3
- `--batch_size`: Default=128
- `--device`: cpu|cuda. Default="cuda"