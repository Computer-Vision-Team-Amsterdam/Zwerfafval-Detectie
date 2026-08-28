import argparse
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy.typing as npt
from output_image import OutputImage
from tqdm.auto import tqdm

WINDOW_NAME = "Visualize YOLO predictions"

CLASS_TO_NAME = {
    0: "Zwerfafval_grof",
    1: "Zwerfafval_fijn",
    2: "Grofvuil",
    3: "Vuilniszak",
}

CLASS_TO_COLOR = {
    0: (140, 0, 227),
    1: (242, 188, 0),
    2: (90, 20, 40),
    3: (43, 231, 251),
}

STATE_PREDICTIONS = 0
STATE_LABELS = 1


MIN_CONF = 0.3
CLASSES_TO_DRAW = [0]


class PredictionDrawer:
    """
    Visualization tool for YOLO predictions. The tool shows predictions (and
    optionally ground truth labels) on images and allows to filter by confidence
    score.

    Parameters
    ----------
    images_folder: str
        Path to folder containing the images.
    predictions_folder: str
        Path to folder containing the YOLO predictions.
    labels_folder: Optional[str] = None
        Optional: path to folder containing the ground truth labels.
    """

    raw_image: npt.NDArray
    pred_image: OutputImage

    min_confidence = 0.0
    current_state = STATE_PREDICTIONS

    def __init__(
        self,
        images_folder: str,
        predictions_folder: str,
        output_folder: Optional[str] = None,
    ):
        self.images_folder = images_folder
        self.predictions_folder = predictions_folder
        self.output_folder = output_folder

        os.makedirs(output_folder, exist_ok=True)

        self._scan_folders()
        self._draw_predictions()

    def _scan_folders(self) -> None:
        print("Scanning input folders...")

        self.image_files = get_file_paths(self.images_folder, [".jpg", ".jpeg", ".png"])
        print(f" - Found {len(self.image_files)} image files")

        self.prediction_files = set(get_file_paths(self.predictions_folder, [".txt"]))
        print(f" - Found {len(self.prediction_files)} prediction files")

    def _save_image(self, name: str) -> None:
        """Show the image."""
        out_file = os.path.join(self.output_folder, name)
        cv2.imwrite(out_file, self.pred_image.get_image())

    def _draw_predictions(self) -> None:
        """Main window and logic."""
        for img_file in tqdm(self.image_files):
            pred_file = f"{os.path.splitext(img_file)[0]}.txt"
            img_file_path = os.path.join(self.images_folder, img_file)

            raw_image = cv2.imread(img_file_path)

            predictions: List[Dict[str, Any]] = []

            if pred_file in self.prediction_files:
                pred_file_path = os.path.join(self.predictions_folder, pred_file)
                predictions = load_yolo_annotations(
                    annotation_file=pred_file_path, class_ids=CLASSES_TO_DRAW
                )

            self.pred_image = generate_image(raw_image, predictions, MIN_CONF)

            self._save_image(name=img_file)

        print("All images saved.")
        print("Exiting")
        return


def generate_image(
    raw_image: npt.NDArray, detections: List[Dict[str, Any]], min_conf: float = 0.0
):
    """
    Draw legend and annotations on the image for the given list of detections
    and confidence threshold.
    """
    image = OutputImage(raw_image.copy())
    img_width, img_height = image.shape[1], image.shape[0]

    if min_conf > 0.0:
        filtered_detections = [det for det in detections if det["conf"] >= min_conf]
    else:
        filtered_detections = detections

    if len(filtered_detections) > 0:
        cats = sorted(set([dets["cls"] for dets in detections]))
        image.draw_legend(
            origin=(img_width - 100, 100),
            categories=cats,
            category_names=CLASS_TO_NAME,
            colour_map=CLASS_TO_COLOR,
        )

    for pred in filtered_detections:
        bbox = convert_yolo_bbox_for_img(
            bbox=pred["bbox"], img_w=img_width, img_h=img_height
        )
        obj_class = pred["cls"]
        text = f"{pred['conf']:.2f}"

        image.draw_bounding_boxes(
            boxes=[bbox],
            categories=[obj_class],
            color_map=CLASS_TO_COLOR,
            texts=[text],
            line_thickness=1,
            font_scale=0.5,
            font_thickness=1,
        )

    return image


def get_file_paths(
    folder: str,
    file_type: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    """
    List all files with a given file_type (default: .json) in folder. Returns a
    sorted list.
    """
    if file_type is None:
        files = os.listdir(folder)
    else:
        if isinstance(file_type, str):
            file_type = [file_type]
        files = [
            file
            for file in os.listdir(folder)
            if os.path.splitext(file)[1] in file_type
        ]
    return sorted(files)


def load_yolo_annotations(
    annotation_file: str, class_ids: Optional[List[int]] = None
) -> List[dict[str, Any]]:
    """
    Load YOLO annotations from file and return them as dict.
    """
    annot_dicts = []
    with open(annotation_file, "r") as f:
        for line in f.readlines():
            parts = line.split(sep=" ")
            if len(parts) == 7:
                cls_id, x_center, y_center, w, h, conf, _ = map(
                    float, line.split(sep=" ")
                )
                is_pred = True
            else:
                cls_id, x_center, y_center, w, h = map(float, line.split(sep=" "))
                conf = 1.0
                is_pred = False
            if (class_ids is None) or (cls_id in class_ids):
                annot_dicts.append(
                    {
                        "is_prediction": is_pred,
                        "cls": int(cls_id),
                        "bbox": {
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": w,
                            "height": h,
                        },
                        "conf": conf,
                    }
                )
    return annot_dicts


def convert_yolo_bbox_for_img(
    bbox: dict[str, float], img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """
    Convert YOLO bounding box to image coordinates for drawing.
    """
    x_min = bbox["x_center"] - bbox["width"] / 2
    x_max = bbox["x_center"] + bbox["width"] / 2
    y_min = bbox["y_center"] - bbox["height"] / 2
    y_max = bbox["y_center"] + bbox["height"] / 2
    x_min = int(x_min * img_w)
    x_max = int(x_max * img_w)
    y_min = int(y_min * img_h)
    y_max = int(y_max * img_h)

    return (x_min, y_min, x_max, y_max)


def main():
    parser = argparse.ArgumentParser(
        description="""
            Display YOLO predictions, optionally compared to ground truth.\n
            Specify paths to images, prediction labels, and optional ground truth labels.
        """,
    )
    parser.add_argument(
        "--images_folder",
        type=str,
        required=True,
        help="e.g. /home/user/dataset/images",
    )
    parser.add_argument(
        "--predictions_folder",
        type=str,
        required=True,
        help="e.g. /home/user/dataset/predictions",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="e.g. /home/user/dataset/draw_predictions",
    )
    args = parser.parse_args()

    images_folder = args.images_folder
    predictions_folder = args.predictions_folder
    output_folder = args.output_folder

    _ = PredictionDrawer(images_folder, predictions_folder, output_folder)


if __name__ == "__main__":
    main()
