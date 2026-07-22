from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from zwerfafval_detectie.utils_eval import read_annotations_folder


class Evaluator:
    """
    Evaluator for model prediction results. The methods are specifically
    designed for "zwerfafval" (litter) objects in street view images, which are
    assumed to be small in size w.r.t. the image. We care less about predicting
    the exact bounding box of each object but only check whether it's centroid
    is covered by a prediction (and vice versa).

    Specifically:
    - We measure _recall_ based on whether the centroid of a ground truth
    bounding box is covered by a prediction bounding box.
    - We measure _precision_ based on whether a prediction bounding box includes
    the centroid of a ground truth bounding box.

    The Evaluator provides methods to compute the precision/recall curve,
    F-score curve, confusion matrix, and object counts per image.

    Parameters
    ----------
    predictions_folder: str
        File path to folder with YOLO-style prediction files.
    labels_folder: str
        File path to folder with YOLO-style ground truth label files.
    categories: Dict[int, str] = {0: "Zwerfafval (grof)", 1: "Zwerfafval (fijn)"}
        Categories to consider, will be used to name columns and possibly filter
        data if the categories provided here are a subset of the model
        predictions and labels.
    confidence: float = 0.0
        Confidence threshold to filter by when computing the confusion matrix
        and object counts.
    """

    DEFAULT_CATEGORIES = {0: "Zwerfafval (grof)", 1: "Zwerfafval (fijn)"}
    DEFAULT_CONFIDENCE = 0.0

    def __init__(
        self,
        predictions_folder: str,
        labels_folder: str,
        categories: Dict[int, str] = DEFAULT_CATEGORIES,
        confidence: float = DEFAULT_CONFIDENCE,
    ):
        self.categories = categories
        self.confidence = confidence

        self.dummy_series = pd.Series(
            data={cat_id: 0.0 for cat_id in self.categories.keys()}
        )
        self.categories_extra: Dict[Union[int, str], str] = {
            key: value for key, value in self.categories.items()
        }
        self.categories_extra["total"] = "Totaal"

        self.labels_gdf = read_annotations_folder(
            folder_path=labels_folder,
            categories=self.categories.keys(),
        )
        self.predictions_gdf = read_annotations_folder(
            folder_path=predictions_folder,
            categories=self.categories.keys(),
        )

        self._prep_precision_recall()

    def get_predictions(self) -> gpd.GeoDataFrame:
        """Get the predictions as GeoDataFrame."""
        return self.predictions_gdf

    def get_labels(self) -> gpd.GeoDataFrame:
        """Get the ground truth labels as GeoDataFrame."""
        return self.labels_gdf

    def _prep_precision_recall(self) -> None:
        """Prepare precision and recall data for later use."""
        _recall_dfs: List[pd.DataFrame] = []

        for cat in self.categories.keys():
            _labels_tmp = self.labels_gdf[self.labels_gdf["category"] == cat]
            _pred_tmp = self.predictions_gdf[self.predictions_gdf["category"] == cat]
            _recall_dfs.append(
                (
                    _labels_tmp.set_geometry(_labels_tmp.centroid)
                    .sjoin(
                        _pred_tmp,
                        how="left",
                        predicate="covered_by",
                        on_attribute="file_name",
                    )
                    .reset_index()
                    .sort_values(by="confidence_right", ascending=False)
                    .drop_duplicates(subset="index")
                    .set_index("index")
                    .sort_index()
                )
            )

        self.recall_df = pd.concat(_recall_dfs).sort_index()

        _precision_df = self.predictions_gdf.sjoin(
            self.labels_gdf.set_geometry(self.labels_gdf.centroid),
            how="left",
            predicate="contains",
            on_attribute="file_name",
        )
        _precision_df["match_cat"] = (
            _precision_df["category_left"] == _precision_df["category_right"]
        )
        self.precision_df = pd.DataFrame(
            _precision_df.reset_index()
            .sort_values(by="match_cat", ascending=False)
            .drop_duplicates(subset="index")
            .drop(columns="match_cat")
            .set_index("index")
            .sort_index()
        )

    def compute_recall(
        self, confidence_threshold: float = DEFAULT_CONFIDENCE
    ) -> pd.Series:
        """
        Get the recall for a specific confidence threshold.
        """
        gt_total = self.recall_df.value_counts(subset="category_left")

        _temp: pd.Series = self.recall_df[
            (self.recall_df["confidence_right"] >= confidence_threshold)
            & (self.recall_df["category_right"] == self.recall_df["category_left"])
        ].value_counts(subset="category_left")
        pred_total = self.dummy_series.copy()
        pred_total.loc[_temp.index] = _temp

        recall_overall = (
            pred_total.sum() / gt_total.sum() if gt_total.sum() > 0 else np.nan
        )

        recall = pred_total / gt_total
        recall["total"] = recall_overall
        return recall

    def compute_precision(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
    ) -> pd.Series:
        """
        Get the precision for a specific confidence threshold.
        """
        _precision_df = self.precision_df[
            self.precision_df["confidence_left"] >= confidence_threshold
        ]
        pred_total = _precision_df.value_counts(subset="category_left")

        _temp: pd.Series = _precision_df[
            (_precision_df["confidence_right"] == 1.0)
            & (_precision_df["category_right"] == _precision_df["category_left"])
        ].value_counts(subset="category_left")
        gt_total = self.dummy_series.copy()
        gt_total.loc[_temp.index] = _temp

        precision_overall = (
            gt_total.sum() / pred_total.sum() if pred_total.sum() > 0 else np.nan
        )

        precision = gt_total / pred_total
        precision["total"] = precision_overall
        return precision

    def get_precision_recall_stats(self) -> pd.DataFrame:
        """
        Compute precision and recall, for each category and as a total, for a
        range of confidence thresholds and return the results as a DataFrame.
        """

        conf_values = np.arange(0.0, 1.01, 0.05)

        data = {
            "confidence": conf_values,
        }

        precision: List[pd.Series] = [
            self.compute_precision(confidence_threshold=c) for c in conf_values
        ]
        recall: List[pd.Series] = [
            self.compute_recall(confidence_threshold=c) for c in conf_values
        ]

        for cat_id, cat_name in self.categories_extra.items():
            data[f"Precision - {cat_name}"] = [
                result.loc[cat_id] for result in precision
            ]
            data[f"Recall - {cat_name}"] = [result.loc[cat_id] for result in recall]

        self.precision_recall_df = pd.DataFrame(data=data).set_index("confidence")
        return self.precision_recall_df

    def get_f_score(self, beta: float = 1.0) -> pd.DataFrame:
        """
        Compute `f_beta-score` for a chosen version of beta (e.g. f1 score when
        beta=1). Uses the result of `get_precision_recall_stats()` as input;
        will call this method if it hasn't been called yet. Returns the F-score
        over a range of confidence thresholds as DataFrame.
        """

        if not hasattr(self, "precision_recall_df"):
            print("Calling get_precision_recall_stats() first...")
            self.get_precision_recall_stats()

        _cats_p = [
            "Precision - Totaal",
            "Precision - Zwerfafval (grof)",
            "Precision - Zwerfafval (fijn)",
        ]
        _cats_r = [
            "Recall - Totaal",
            "Recall - Zwerfafval (grof)",
            "Recall - Zwerfafval (fijn)",
        ]

        f_score: pd.DataFrame = (
            (1 + np.power(beta, 2))
            * self.precision_recall_df[_cats_p].mul(
                self.precision_recall_df[_cats_r].values
            )
        ) / (
            np.power(beta, 2)
            * self.precision_recall_df[_cats_p].add(
                self.precision_recall_df[_cats_r].values
            )
        )
        f_score.columns = f_score.columns.str.replace("Precision", "F-score")
        return f_score

    def get_confusion_matrix(
        self, confidence_threshold: float = DEFAULT_CONFIDENCE
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the confusion matrix and the normalized confusion matrix for a given
        confidence threshold as DataFrames.
        """
        conf_precision_df = (
            self.precision_df[
                self.precision_df["confidence_left"] >= confidence_threshold
            ][["category_left", "category_right"]]
            .fillna("Background")
            .groupby("category_left")
            .value_counts()
            .unstack()
            .rename_axis(None)
            .rename_axis(None, axis=1)
            .rename(index=self.categories, columns=self.categories)
        )

        conf_recall_df = (
            self.recall_df[
                self.recall_df["category_right"].isna()
                | (self.recall_df["confidence_right"] <= confidence_threshold)
            ][["category_left", "category_right"]]
            .fillna("Background")
            .groupby("category_left")
            .count()
            .transpose()
            .rename_axis(None)
            .rename_axis(None, axis=1)
            .rename(columns=self.categories)
            .rename(index={"category_right": "Background"})
        )

        columns = ["Zwerfafval (grof)", "Zwerfafval (fijn)"]
        if "Background" in conf_precision_df.columns:
            columns.append("Background")

        conf_df = pd.concat((conf_precision_df, conf_recall_df))[columns]
        conf_df.index = pd.MultiIndex.from_product([["Prediction"], conf_df.index])
        conf_df.columns = pd.MultiIndex.from_product(
            [["Ground Truth"], conf_df.columns]
        )

        conf_df_normalized = conf_df.div(conf_df.sum(axis=0), axis=1)

        return conf_df, conf_df_normalized

    def get_counts(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        class_agnostic: bool = False,
    ) -> pd.DataFrame:
        """
        Get the object counts per image for a given confidence threshold as
        DataFrame. Set `class_agnostic` to True to count all objects together
        instead of per category.
        """
        labels_sorted = self.labels_gdf.set_index("file_name").sort_index()
        preds_sorted = (
            self.predictions_gdf[
                self.predictions_gdf["confidence"] >= confidence_threshold
            ]
            .set_index("file_name")
            .sort_index()
        )

        if class_agnostic:
            cat_map = {0: "Zwerfafval", 1: "Zwerfafval"}
        else:
            cat_map = self.categories

        labels_category_counts = (
            labels_sorted[["category"]]
            .replace(cat_map)
            .groupby(["file_name", "category"])
            .size()
            .unstack(fill_value=0)
        )
        preds_category_counts = (
            preds_sorted[["category"]]
            .replace(cat_map)
            .groupby(["file_name", "category"])
            .size()
            .unstack(fill_value=0)
        )

        merged = labels_category_counts.join(
            other=preds_category_counts,
            how="outer",
            lsuffix=" (true)",
            rsuffix=" (pred)",
        )

        return merged.fillna(0)
