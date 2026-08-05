import os

from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from zwerfafval_detectie import aml_interface, settings
from zwerfafval_detectie.inference_pipeline.components.run_inference import (
    run_inference,
)
from zwerfafval_detectie.training_pipeline.components import train_model


@pipeline()
def training_pipeline():

    datastore_path = aml_interface.get_datastore_full_path(
        settings["training_pipeline"]["inputs"]["datastore_path"]
    )
    training_rel_path = settings["training_pipeline"]["inputs"][
        "training_data_rel_path"
    ]
    model_weights_rel_path = settings["training_pipeline"]["inputs"][
        "model_weights_rel_path"
    ]
    project_datastore_path = aml_interface.get_datastore_full_path(
        settings["training_pipeline"]["outputs"]["project_datastore_path"]
    )
    project_rel_path = settings["training_pipeline"]["outputs"]["project_rel_path"]

    training_data_path = os.path.join(datastore_path, training_rel_path)
    training_data = Input(
        type=AssetTypes.URI_FOLDER,
        path=training_data_path,
    )

    model_weights_path = os.path.join(datastore_path, model_weights_rel_path)
    model_weights = Input(
        type=AssetTypes.URI_FOLDER,
        path=model_weights_path,
    )

    train_model_step = train_model(
        mounted_dataset=training_data, model_weights=model_weights
    )
    train_model_step.outputs.yolo_yaml_path = Output(
        type="uri_folder", mode="rw_mount", path=model_weights_path
    )

    project_path = os.path.join(project_datastore_path, project_rel_path)
    train_model_step.outputs.project_path = Output(
        type="uri_folder", mode="rw_mount", path=project_path
    )

    if settings["training_pipeline"]["run_prediction_best_model"]:
        inference_data_path = os.path.join(training_data_path, "images")
        inference_data = Input(
            type=AssetTypes.URI_FOLDER,
            path=inference_data_path,
        )

        model_weights_rel_path = os.path.join(
            settings["training_pipeline"]["outputs"]["experiment_name"],
            "weights",
            "best.pt",
        )

        run_inference_step = run_inference(
            inference_data_dir=inference_data,
            model_weights_dir=train_model_step.outputs.project_path,
            model_weights_forced_path=model_weights_rel_path,
        )

        output_path = os.path.join(
            project_path,
            settings["training_pipeline"]["outputs"]["experiment_name"],
            "predict",
        )
        run_inference_step.outputs.output_dir = Output(
            type="uri_folder", mode="rw_mount", path=output_path
        )

    return {}


def main() -> None:
    aml_interface.submit_pipeline_experiment(
        pipeline_function=training_pipeline,
        experiment_name=settings["aml_experiment_details"]["experiment_name"],
        default_compute=settings["aml_experiment_details"]["compute_name"],
        show_log=False,
    )


if __name__ == "__main__":
    main()
