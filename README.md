# Music Composition with Linear Transformers

## Install Requirements

`pip install -r requirements.txt`

## 1. Data Download
### 1.1 Download VGMIDI dataset
Clone the VGMIDI dataset repository, go to the unabelled pieces directory and
extract the `midi.zip` file:

```
git clone
cd vgmidi/unlabelled
unzip midi.zip
```

### 1.2 Split the dataset
Go to the `src` directory of unabelled pieces and run the `midi_split` script:

```
cd unlabelled/src
python3 midi_split.py --csv ../../vgmidi_unlabelled.csv --midi ../midi
```

## 2. Data Pre-processing
### 2.1 Data Augmentation
Augment the VGMIDI dataset with tranposition and time strech as defined by [Oore et al. 2017]():

`python3 augment.py --midi <path_to_midi_data>`

### 2.2 Data Encoding
Encode the VGMIDI dataset with the encoding scheme as defined by [Oore et al. 2017]():

`python3 encoder.py --midi <path_to_midi_data>`

<!-- ## Train -->

## 3. Train Model
Train a [fast-tranformer]() on the VGMIDI dataset:

`python3 train.py --train vgmidi_augmented/train/ --test vgmidi_augmented/test --seq_len 2048 --lr 1e-04 --epochs 50 --d_query 32 --n_heads 8 --n_layers 8 --batch_size 16 --save_to trained/model_{}.pth`

## 4. Generate Music
This repository includes a pre-trained model (`trained/model40.pth`). To generate music with that model:

`python3 generate.py --seq_len 2048 --n_layers 8 --model trained/model40.pth --vocab_size 390 --k 45 --p 0.9`

To generate music with a model you trained yourself, simply change the --model argument to point to the model saved during training.
