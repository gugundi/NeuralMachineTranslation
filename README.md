# Neural Machine Translation
An implementation of a neural machine translation system with a LSTM encoder-decoder plus attention architecture.

# Authors
The authors of this project is Gabriel Kihembo Enemark-Broholm, Christoffer Øhrstrøm and Oldouz Majidi.

# Results
We managed to get a BLEU score of 47 on the Multi30k dataset

## Recreating results
Run the program with configuration (final.json)

*python main.py --config final.json*

# How to run
Navigate to the root folder of the project and run the following command in terminal/cmd:

*python main.py*

The following arguments may then be added:

* --config_path
  + For specifying the path to the configurations folder
* --load_weights
  + For loading previously stored weights of model (boolean)
* --config
  + Path to model configuration (defaults to 'configs/default.json')
* --debug
  + Debug mode
* --dummy_fixed_length
  + Use a dummy dataset of fixed length sentences
* --dummy_variable_length
  + Use a dummy dataset of variable length sentences
* --iwslt
  + Use the IWSLT dataset
* --name
  + Name used when writing to tensorboard (visualiation)

The default dataset is the Multi30K dataset.


# Dependencies
* Python 3.6.5
* Run script hpc/install_requirements.sh
