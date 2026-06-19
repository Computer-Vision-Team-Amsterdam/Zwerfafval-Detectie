import argparse
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy.typing as npt
from output_image import OutputImage

WINDOW_NAME = "Visualize YOLO predictions"

CLASS_TO_NAME = {
    0: "Zwerfafval_grof",
    1: "Zwerfafval_fijn",
    2: "Grofvuil",
}

CLASS_TO_COLOR = {
    0: (140, 0, 227),
    1: (242, 188, 0),
    2: (90, 20, 40),
}

STATE_PREDICTIONS = 0
STATE_LABELS = 1


class PredictionVisualizer:
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
    labels: List[Dict[str, Any]]
    predictions: List[Dict[str, Any]]

    min_confidence = 0.0
    current_state = STATE_PREDICTIONS

    def __init__(
        self,
        images_folder: str,
        predictions_folder: str,
        labels_folder: Optional[str] = None,
    ):
        self.images_folder = images_folder
        self.predictions_folder = predictions_folder
        self.labels_folder = labels_folder

        print(
            "\n======\n"
            "Visualize YOLO predictions.\n"
            "Press [RIGHT] or [LEFT] to move between images.\n"
            "Press [UP] and [DOWN] to toggle between predictions and truth labels (if available).\n"
            "Press [ESC] to exit."
            "\n======\n"
        )

        self._scan_folders()
        self._visualize_predictions()

    def _scan_folders(self) -> None:
        print("Scanning input folders...")

        self.image_files = get_file_paths(self.images_folder, [".jpg", ".jpeg", ".png"])
        print(f" - Found {len(self.image_files)} image files")

        self.prediction_files = set(get_file_paths(self.predictions_folder, [".txt"]))
        print(f" - Found {len(self.prediction_files)} prediction files")

        if self.labels_folder is not None:
            self.label_files = set(get_file_paths(self.labels_folder, [".txt"]))
            print(f" - Found {len(self.label_files)} label files")
        else:
            self.label_files = set()

    def change_confidence(self, new_value: int) -> None:
        """
        Update confidence score threshold, re-draw the image, and refresh the
        display.
        """
        self.min_confidence = new_value / 100
        self._refresh_images()
        self._show_image()

    def _refresh_images(self) -> None:
        """
        Re-draw images (e.g. when moving to the next image or when changing the
        confidence threshold).
        """
        self.pred_image = generate_image(
            self.raw_image, self.predictions, self.min_confidence
        )
        self.label_image = generate_image(self.raw_image, self.labels)

    def _show_image(self) -> None:
        """Show the image."""
        if self.current_state == STATE_PREDICTIONS:
            cv2.setWindowTitle(
                WINDOW_NAME,
                f"PREDICTIONS for image {self.image_file} (img {self.idx}/{self.n_images})",
            )
            cv2.imshow(WINDOW_NAME, self.pred_image.get_image())
        elif self.current_state == STATE_LABELS:
            cv2.setWindowTitle(
                WINDOW_NAME,
                f"LABELS for image {self.image_file} (img {self.idx}/{self.n_images})",
            )
            cv2.imshow(WINDOW_NAME, self.label_image.get_image())
        else:
            print(f"Illegal state: {self.current_state}")

    def _visualize_predictions(self) -> None:
        """Main window and logic."""
        cv2.namedWindow(WINDOW_NAME)
        cv2.createTrackbar(
            "Confidence score", WINDOW_NAME, 0, 100, self.change_confidence
        )

        self.n_images = len(self.image_files)
        self.idx = 0

        while self.idx < self.n_images:
            self.image_file = self.image_files[self.idx]
            file_name = os.path.splitext(self.image_file)[0]
            img_file_path = os.path.join(self.images_folder, self.image_file)

            self.raw_image = cv2.imread(img_file_path)

            if f"{file_name}.txt" in self.prediction_files:
                pred_file_path = os.path.join(
                    self.predictions_folder, f"{file_name}.txt"
                )
                self.predictions = load_yolo_annotations(pred_file_path)
            else:
                print(f"No predictions found for image {self.image_file}")
                self.predictions = []

            self.labels = []
            if self.labels_folder is not None:
                if f"{file_name}.txt" in self.label_files:
                    label_file_path = os.path.join(
                        self.labels_folder, f"{file_name}.txt"
                    )
                    self.labels = load_yolo_annotations(label_file_path, is_pred=False)
                else:
                    print(f"No labels found for image {self.image_file}")

            self._refresh_images()
            self._show_image()

            k = None
            while True:
                k = cv2.waitKey(0)  # Waits for a key event infinitely (when delay<=0)
                if k in [44]:  # [<]: previous image
                    self.idx -= 1
                    break
                elif k in [46]:  # [right]: next image
                    self.idx += 1
                    break
                elif k in [108]:  # [L]: show label image
                    if self.labels_folder is not None:
                        self.current_state = STATE_LABELS
                        self._show_image()
                    else:
                        pass
                elif k in [112]:  # [P]: show prediction image
                    self.current_state = STATE_PREDICTIONS
                    self._show_image()
                elif k == 27:  # [esc]: exit the program
                    print("Exiting")
                    cv2.destroyWindow(WINDOW_NAME)
                    return
                else:
                    print("Key not valid...")

        print("All images viewed.")
        print("Exiting")
        cv2.destroyWindow(WINDOW_NAME)
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
            font_scale=0.3,
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
    annotation_file: str, is_pred: bool = True
) -> List[dict[str, Any]]:
    """
    Load YOLO annotations from file and return them as dict.
    """
    annot_dicts = []
    with open(annotation_file, "r") as f:
        for line in f.readlines():
            if is_pred:
                cls_id, x_center, y_center, w, h, conf, _ = map(
                    float, line.split(sep=" ")
                )
            else:
                cls_id, x_center, y_center, w, h = map(float, line.split(sep=" "))
                conf = 1.0
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
        "--labels_folder",
        type=str,
        required=False,
        help="e.g. /home/user/dataset/labels",
    )
    args = parser.parse_args()

    images_folder = args.images_folder
    predictions_folder = args.predictions_folder
    labels_folder = args.labels_folder

    _ = PredictionVisualizer(images_folder, predictions_folder, labels_folder)


if __name__ == "__main__":
    main()
