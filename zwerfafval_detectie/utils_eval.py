import os
from typing import Any, Dict, Iterable, List, Optional

import geopandas as gpd
import pandas as pd
import shapely.geometry as sg
from pandas.io.formats.style import Styler


def yolo_string_to_dict(yolo_annotation: str) -> Dict[str, Any]:
    """
    Convert a line of YOLO annotation text to a dict format.

    Expects `cat_id x_center y_center width height conf`.

    Returns `{"category": int, "confidence": float, "geometry": shapely.geometry.box}`.
    """
    values = yolo_annotation.split()

    if len(values) >= 6:
        cat_id, x_center, y_center, width, height, conf = map(float, values[:6])
    elif len(values) == 5:
        cat_id, x_center, y_center, width, height = map(float, values)
        conf = 1.0
    else:
        raise ValueError(f"Unrecognized annotation string format: {yolo_annotation}")

    bbox = sg.box(
        minx=x_center - width / 2,
        miny=y_center - height / 2,
        maxx=x_center + width / 2,
        maxy=y_center + height / 2,
    )

    data = {"category": int(cat_id), "confidence": conf, "geometry": bbox}

    return data


def yolo_file_to_dicts(annotation_file_path: str) -> List[Dict[str, Any]]:
    """
    Read a YOLO annotations file and return the annotations as a list of dicts.

    See `yolo_string_to_dict(..)` for details on the dict contents.
    """
    data = []

    with open(annotation_file_path, "r") as f:
        file_name = os.path.basename(annotation_file_path)
        for line in f.readlines():
            line_data = yolo_string_to_dict(line)
            line_data["file_name"] = file_name
            data.append(line_data)

    return data


def read_annotations_folder(
    folder_path: str, categories: Optional[Iterable[int]], agnostic: bool = False
) -> gpd.GeoDataFrame:
    """
    Convert all YOLO annotation files in a folder to GeoDataFrame with one annotation per row.

    The GeoDataFrame has columns `"file_name", "category", "confidence", "geometry"`.
    """
    data = []
    annotation_files = [
        file for file in os.listdir(folder_path) if os.path.splitext(file)[1] == ".txt"
    ]

    for file in annotation_files:
        data.extend(yolo_file_to_dicts(os.path.join(folder_path, file)))

    gdf = gpd.GeoDataFrame(
        data=data, columns=["file_name", "category", "confidence", "geometry"]
    )

    if categories is not None:
        gdf = gdf[gdf["category"].isin(categories)]

    if agnostic:
        gdf["category"] == 0

    return gdf


def make_pretty_confusion_matrix(
    df: pd.DataFrame, precision: int = 0, title: str = "Confusion Matrix"
) -> Styler:
    """
    Prettify a confusion matrix to mimic the style of YOLO output after training.
    """
    styles = [
        dict(
            selector="*", props=[("font-family", "sans-serif"), ("font-size", "16px")]
        ),
        dict(selector="th", props=[("text-align", "center")]),
        dict(
            selector="th.row_heading.level0",
            props=[
                ("transform", "rotate(270deg);"),
                ("height", "250px"),
                ("font-weight", "bold"),
            ],
        ),
        dict(
            selector="th.col_heading.level0",
            props=[("width", "250px"), ("font-weight", "bold")],
        ),
        dict(
            selector="caption", props=[("font-size", "20px"), ("font-weight", "bold")]
        ),
    ]

    return (
        df.style.set_caption(title)
        .background_gradient(axis=None, vmin=0, cmap="Blues")
        .highlight_null(color="white", subset=None, props=None)
        .format(na_rep="", precision=precision)
        .set_table_styles(styles)
        .set_properties(**{"text-align": "center"})
    )
