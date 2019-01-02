# Neural Machine Translation
An implementation of a neural machine translation system with a LSTM encoder-decoder plus attention architecture.

# Authors
The authors of this project are Gabriel Kihembo Enemark-Broholm, Christoffer Øhrstrøm and Oldouz Majidi.

# Results
We got a BLEU score of **30.37** using local attention versus **16.62** with no attention mechanism.

## Recreating the results
To recreate and visualize the results run the Jupiter notebook *test.ipynb*

# Training the model
To train the model using the settings we used run the program with configuration final.json.

*python main.py --config final.json*

The following arguments may then be used to configure the model:

* --config
  + Path to model configuration (defaults to 'configs/default.json')
* --debug
  + Use a debug dataset.
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
