# Litter Detection

Detection of litter on streetview images. In this project we train a YOLO model on a dataset of annotated images.


## Installation

See [instructions here](INSTALL.md).


## Preparing the dataset

See the notebooks for [data selection](notebooks/zwerfafval_dataselectie.ipynb) and [creating a train/val/test split](notebooks/training_set.ipynb).


## Creating the AzureML environment

Run `az login`, and then create the environment from the [Dockerfile](zwerfafval_detectie/create_aml_environment/Dockerfile) using:

```shell
uv run --extra dev python zwerfafval_detectie/create_aml_environment/create_azure_env.py
```


## Training a model

The training script provided is meant to be run on AzureML. A selection of training parameters can be configured in [config.yml](config.yml) under `training_pipeline`, with any additional parameters through the optional [model_parameters.json](model_parameters.json).

Make sure your environment contains a config.json file with the appropriate AzureML configuration:

```json
{
    "subscription_id": "...",
    "resource_group": "...",
    "workspace_name": "..."
}
```

Run `az login`, and then start a training run with the current configuration:

```shell
uv run --extra dev python zwerfafval_detectie/training_pipeline/submit_training_pipeline.py
```

### Model weights

Our current model is a prototype trained on a limited set of images with two classes:
```python
{
    0: "zwerfafval_grof",  # litter bigger than 10cm, but not more than a typical piece of cardboard or plastic bag
    1: "zwerfafval_fijn"   # litter small than 10cm
}
```
The model is based on the [pre-trained YOLO26m](https://github.com/ultralytics/ultralytics#-models) with an image size of 1920. The weights can be found [here](model_weights/best.pt).


## Running inference on (new) data

Inference parameters can be configured in [config.yml](config.yml) under `inference_pipeline`.

```shell
uv run --extra dev python zwerfafval_detectie/inference_pipeline/submit_inference_pipeline.py
```


## Visualising predictions

The script [visualize_predictions.py](scripts/visualize_predictions.py) can be used to check the model predictions on a set of images, along with optional ground truth labels.

```shell
uv run --extra dev python scripts/visualize_predictions.py --images_folder /path/to/images --predictions_folder /path/to/predictions [--labels_folder /path/to/labels]
```


## Model evaluation

The [evaluation notebook](notebooks/evaluate.ipynb) can be used to measure the performance of a trained model on a validation or test set.
