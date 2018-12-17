# Neural Machine Translation
An implementation of a neural machine translation system with a LSTM encoder-decoder plus attention architecture.

# Authors
The authors of this project are Gabriel Kihembo Enemark-Broholm, Christoffer Øhrstrøm and Oldouz Majidi.

# Results
Below is a list of the result we got using different n-gram BLEU score evaluations:
* 1-gram: 56.37
* 2-gram: 40.74
* 3-gram: 30.04
* 4-gram: 22.61

## Recreating the results
To recreate and visualize the results run the Jupiter notebook *test.ipynb*

## Training the model
To train the model using the settings we used run the program with configuration final.json.

*python main.py --config final.json*

# How to run
Navigate to the root folder of the project and run the following command in terminal/cmd:

*python main.py*

The following arguments may then be added:

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
