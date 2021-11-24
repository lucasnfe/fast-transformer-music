# Music Composition with Linear Transformers

## Install Requirements

`pip install -r requirements.txt`

## Data Download and Pre-processing

### 1. Download VGMIDI dataset

Clone the VGMIDI dataset repository, go to the unabelled pieces directory and
extract the `midi.zip` file:

```
git clone
cd vgmidi/unlabelled
unzip midi.zip
```

### 2. Split the dataset

Go to the `src` directory of unabelled pieces and run the `midi_split` script:

```
cd unlabelled/src
python3 midi_split.py --csv ../../vgmidi_unlabelled.csv --midi ../midi
```

### 3. Data Augmentation

Augment the VGMIDI dataset with tranposition and time strech as defined by [Oore et al. 2017]():

`python3 augment.py --midi <path_to_midi_data>`

### 4. Data Encoding

Encode the VGMIDI dataset with the encoding scheme as defined by [Oore et al. 2017]():

`python3 encoder.py --midi <path_to_midi_data>`

<!-- ## Train -->
