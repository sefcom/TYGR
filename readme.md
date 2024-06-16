# TYGR

Type Inference on Stripped Binaries using Graph Neural Networks

## Setup

Make sure you install `torch`, `torch_geometric`, `pyvex`, and `elftools`.
It is recommended that you use `conda` to manage your environment.
CUDA is supported during the training and testing process.

To use the provided `environment.yml`, please use

```
conda env create -f environment.yml
```

This will create a conda environment called `TYGR`.
You can then call

```
conda activate TYGR
```

to activate it.
It is expected that you may need to install different versions of the
above libraries (e.g. `torch_geometric`).
In this case, you should manually install the dependencies.

## How to use

### 1. Dataset Generation

To train a new model, first try to generate dataset from binaries:

```
./TYGR datagen PATH/TO/YOUR/BINARY DATASET.pkl
```

Make sure the binaries being used in this stage is properly compiled with DWARF
information available.
If you are compiling using GCC or Clang, you can include such information using
a flag `-gdwarf`.

Note that one can specify a `--method` flag which defaults to `glow`, representing
our best model.
To try out other methods like Transformer or BiLSTM, you should feed another flag
to the datagen process to generate data suitable for the corresponding method.

You can combine different datasets by using the `datamerge` command:

```
./TYGR datamerge DATASET1.pkl DATASET2.pkl -o MERGED.pkl
```

Then you can use split_model_pkl.py in script folder to parse the dataset for MODEL base and MODEL struct.

Lastly, you can split the dataset into training, validation, and testing sets

```
./TYGR datasplit MERGED.pkl --train TRAIN.pkl --validation VALID.pkl --test TEST.pkl
```

By default, we use 80% of the data on training, 10% for validation and 10% for testing.
You can change this by feeding more command line flags.
Please consult `./TYGR datasplit --help` for more information.

### 2. Training & Testing

You can train a new model given that you have the dataset setup.
Then you should use

```
./TYGR train TRAIN.pkl VALID.pkl -o MODEL.model
```

Make sure you feed a desired `--epoch`, `--lr`, and `--method`.
The training won't take very long for our default (best) model.
The best model, as determined by the validation set, will be stored along the way,
named `MODEL.model.best`.
The final model will be stored using the provided name.

To test the trained model, simply do

```
./TYGR test --test TEST.pkl --model MODEL.model
```

where `TEST.pkl` is the testing dataset and the `MODEL.model` is the trained model generated
by the training process.
If `--method` is specified during datagen and training phases, make sure you pass in the same
flag for testing as well.

### 3. Prediction

With a given model you can run prediction on a fresh stripped binary (no need to do datagen).

```
./TYGR predict PATH/TO/YOUR/STRIPPED/BINARY
```

It will try to predict more types than there are, but we will try to cover as many variables
as possible.
In this mode, we identify variables by their function context PC range and stack offset.
