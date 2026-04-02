# OpenVaccine - COVID-19 mRNA Vaccine Degradation Prediction

In 2020, [a Kaggle contest](https://www.kaggle.com/competitions/stanford-covid-vaccine/overview) was held for users to predict the mRNA degradation of potential COVID-19 vaccines.
To get experience applying ML directly to RNA datasets, I decided to participate in the competition and see how well
my results compared to the current leaderboards. 

## Getting started


Clone this repo, create a python virtual environment, and install the package.

```
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

To run pretraining (this runs the BERT masked language modeling task), run the following:

```
openvaccine pretrain
```

To run finetuning (this fine-tunes the model on the stability regression task), run the following:

```
openvaccine finetune
```

To load from a particular checkpoint, pass in the `-c/--checkpoint_dir` argument:

```
openvaccine pretrain -c "outputs/pretrain/checkpoints/30.pth"
```