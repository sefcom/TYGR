# Awesome TYDA data set
Please find TYDA [x86_64 binaries](https://www.dropbox.com/scl/fo/awtitjnc48k224373vcrx/h?rlkey=muj6t1watc6vn2ds6du7egoha&e=1&dl=0). 

This dataset contains C and C++ binaries from Gentoo Linux, each compiled with one of four different compiler optimization flags: O0, O1, O2, and O3. Each set of binaries for a specific optimization level is associated with a .json file named C_CPP_binaries_{opt}.json, where opt represents the optimization level (O0, O1, O2, or O3). These JSON files include metadata about the binary, such as the language, whether it has inline functions or dwarf info, among other details.

Please find the rest of TYDA C binaries here. [c binaries]([https://www.dropbox.com/scl/fo/awtitjnc48k224373vcrx/h?rlkey=muj6t1watc6vn2ds6du7egoha&e=1&dl=0](https://www.dropbox.com/scl/fo/tw64ablil18d7bnzs1ob2/AEYhgKle7gvzdikuCyhKUPE?rlkey=kx4dkj0e4pqam8nqmhq5hl5io&dl=0)).


<br>

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

Please find trained models in `model` folder.

### 1. Dataset Generation

To test a model or train a new model, first try to generate dataset from binaries:

```
./TYGR datagen PATH/TO/YOUR/BINARY DATASET.pkl
```

Make sure the binaries being used in this stage is properly compiled with DWARF
information available.

Note that one can specify a `--method` flag which defaults to `glow`, representing
our best model.
To try out other methods like Transformer or BiLSTM, you should feed another flag
to the datagen process to generate data suitable for the corresponding method.

If the binaries are larger than 1MB I would suggest to run in parallel mode since ANGR is slow. You should be able to specify the number of processors in `src/methods/glow/datagen.py`. Be careful about the memory usage as ANGR may explode your memory. 



You can combine different datasets by using the `datamerge` command:

```
./TYGR datamerge DATASET1.pkl DATASET2.pkl -o MERGED.pkl
```

Then you can use `scripts/split_model_pkl.py` to parse the dataset for MODEL_base and MODEL_struct.

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
./TYGR test MODEL.model TEST.pkl 
```

where `TEST.pkl` is the testing dataset and the `MODEL.model` is the trained model generated
by the training process.
If `--method` is specified during datagen and training phases, make sure you pass in the same
flag for testing as well.

<br>

# Function deduplication

TYDA data set still contains duplicate functions.

Please use the `scripts/angr_func_hash.py` to get the list of duplicate functions, every function in the same hash are duplicate functions.

During `datamerge`, skip functions in the duplicate list.
