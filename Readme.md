Image Caption Generation
-----

An exploration of caption generation for images employing a multi-modal ResNet-LSTM network implemented in PyTorch. The model was trained using the CoCo dataset (https://cocodataset.org/#home) and was evaluated with BLEU scores. To run this model, first collect dataset using `get_datasets.ipynb`, then:

* Define the configuration for your experiment. See `default.json` to see the structure and available options. 
* After defining the configuration (say `my_exp.json`) - run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models will be stored in `./experiment_data/my_exp` dir. Directory storage can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training 
pr evaluate performance.
* If you want to test any model, uncomment the testing line in main.py, be sure to give model_path as an argument to exp.test()
* If you want to train the model with data augmentation, be sure to call CocoDataset with transform = True for training set.

Files
-----
- `main.py` : Main driver class
- `model.py` : Model architecture class
- `experiment.py` : Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging 
and resuming experiments.
- `dataset_factory.py` : Factory to build datasets based on config
- `model_factory.py` : Factory to build models based on config
- `constants.py` : constants used across the project
- `file_utils.py` : utility functions for handling files 
- `caption_utils.py` : utility functions to generate bleu scores
- `vocab.py` : A simple Vocabulary wrapper
- `coco_dataset.py`: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- `get_datasets.ipynb` : A helper notebook to set up the dataset in your workspace
