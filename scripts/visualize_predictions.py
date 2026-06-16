import argparse
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy.typing as npt
from output_image import OutputImage

window_name = "Visualize YOLO predictions"

class_to_name = {
    0: "Zwerfafval_grof",
    1: "Zwerfafval_fijn",
    2: "Grofvuil",
}

class_to_color = {
    0: (140, 0, 227),
    1: (242, 188, 0),
    2: (90, 20, 40),
}


def visualize_predictions(
    images_folder: str, predictions_folder: str, labels_folder: Optional[str] = None
):
    print(
        "\n======\n"
        "Visualize YOLO predictions.\n"
        "Press [RIGHT] or [LEFT] to move between images.\n"
        "Press [UP] and [DOWN] to toggle between predictions and truth labels (if available).\n"
        "Press [ESC] to exit."
        "\n======\n"
    )

    cv2.namedWindow(window_name)

    print("Scanning input folders...")
    image_files = get_file_paths(images_folder, [".jpg", ".jpeg", ".png"])
    print(f" - Found {len(image_files)} image files")
    prediction_files = set(get_file_paths(predictions_folder, [".txt"]))
    print(f" - Found {len(prediction_files)} prediction files")
    if labels_folder is not None:
        label_files = set(get_file_paths(labels_folder, [".txt"]))
        print(f" - Found {len(label_files)} label files")
    else:
        label_files = set()

    n_images = len(image_files)
    idx = 0

    while idx < n_images:
        image_file = image_files[idx]
        file_name = os.path.splitext(image_file)[0]
        img_file_path = os.path.join(images_folder, image_file)

        raw_image = cv2.imread(img_file_path)

        if f"{file_name}.txt" in prediction_files:
            pred_file_path = os.path.join(predictions_folder, f"{file_name}.txt")
            predictions = load_yolo_annotations(pred_file_path)
        else:
            print(f"No predictions found for image {image_file}")
            predictions = []
        pred_image = generate_image(raw_image, predictions)

        labels = []
        if labels_folder is not None:
            if f"{file_name}.txt" in label_files:
                label_file_path = os.path.join(labels_folder, f"{file_name}.txt")
                labels = load_yolo_annotations(label_file_path, is_pred=False)
            else:
                print(f"No labels found for image {image_file}")
            label_image = generate_image(raw_image, labels)

        cv2.setWindowTitle(
            window_name,
            f"Showing predictions for image {image_file} ({idx}/{n_images})",
        )
        cv2.imshow(window_name, pred_image.get_image())

        # The function waitKey waits for a key event infinitely (when delay<=0)
        k = None
        while True:
            k = cv2.waitKey(0)
            if k in [81]:
                # [left]: previous image
                idx -= 1
                break
            elif k in [82]:
                # [up]: show label image
                if labels_folder is not None:
                    cv2.setWindowTitle(
                        window_name,
                        f"Showing labels for image {image_file} ({idx}/{n_images})",
                    )
                    cv2.imshow(window_name, label_image.get_image())
                else:
                    pass
            elif k in [83]:
                # [right]: next image
                idx += 1
                break
            elif k in [84]:
                # [down]: show prediction image
                cv2.setWindowTitle(
                    window_name,
                    f"Showing predictions for image {image_file} ({idx}/{n_images})",
                )
                cv2.imshow(window_name, pred_image.get_image())
            elif k == 27:
                # [esc] to exit the program
                print("Exiting")
                cv2.destroyWindow(window_name)
                return
            else:
                print("Key not valid...")

    print("All images viewed.")
    print("Exiting")
    cv2.destroyWindow(window_name)
    return


def generate_image(raw_image: npt.NDArray, detections: List[Dict[str, Any]]):
    image = OutputImage(raw_image.copy())
    img_width, img_height = image.shape[1], image.shape[0]

    if len(detections) > 0:
        cats = sorted(set([dets["cls"] for dets in detections]))
        image.draw_legend(
            origin=(img_width - 100, 100),
            categories=cats,
            category_names=class_to_name,
            colour_map=class_to_color,
        )

    for pred in detections:
        bbox = convert_yolo_bbox_for_img(
            bbox=pred["bbox"], img_w=img_width, img_h=img_height
        )
        obj_class = pred["cls"]
        # text = class_to_name[obj_class]
        text = f"{pred['conf']:.2f}"

        image.draw_bounding_boxes(
            boxes=[bbox],
            categories=[obj_class],
            colour_map=class_to_color,
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

    Parameters
    ----------
    folder : str
        Folder to list
    file_type : Optional[Union[str, List[str]]] = None
        Optionally filter by (list of) file type

    Returns
    -------
    List[str]
        Sorted list of files
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

    visualize_predictions(images_folder, predictions_folder, labels_folder)


if __name__ == "__main__":
    main()
