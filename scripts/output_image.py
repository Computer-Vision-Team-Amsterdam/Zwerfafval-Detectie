from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt


class OutputImage:
    """
    This class is used to annotate an output image based on model predictions.

    Parameters
    ----------
    image: npt.NDArray
        The original image.
    """

    # Predefined colors for 5 categories
    DEFAULT_COLORS: Dict[int, Tuple[int, int, int]] = {
        0: (255, 0, 0),  # Blue
        1: (0, 255, 0),  # Green
        2: (0, 0, 255),  # Red
        3: (255, 255, 0),  # Cyan
        4: (255, 0, 255),  # Magenta
    }

    def __init__(self, image: npt.NDArray):
        self.image = image
        self.shape = image.shape

    def get_image(self) -> npt.NDArray:
        """Returns the image as Numpy array."""
        return self.image

    def draw_bounding_boxes(
        self,
        boxes: Union[List[Tuple[float, float, float, float]], npt.NDArray[np.float64]],
        categories: Optional[List[int]] = None,
        color_map: Dict[int, Tuple[int, int, int]] = DEFAULT_COLORS,
        box_padding: int = 0,
        line_thickness: int = 3,
        texts: Optional[List[str]] = None,
        font_scale: float = 0.7,
        font_thickness: int = 2,
    ) -> None:
        """
        Draw the given bounding box(es).

        Parameters
        ----------
        boxes : List[Tuple[float, float, float, float]]
            Bounding box(es) to draw, in the format (xmin, ymin, xmax, ymax).
        categories : Optional[List[int]] (default: None)
            Optional: the category of each bounding box. If not provided, colour
            is set to "red".
        color_map : Dict[int, Tuple[int, int, int]]
            Dictionary of BGR colors for each category, in the format
            `{category: (255, 255, 255)}`.
        box_padding : int (default: 0)
            Optional: increase box by this amount of pixels before drawing.
        line_thickness : int (default: 3)
            Line thickness for the bounding box.
        texts : Optional[List[str]] (default: None)
            Optional: list of texts for each bounding box. If not provided, no
            texts are printed.
        font_scale : float (default: 0.7)
            Font scale for the text.
        font_thickness : int (default: 2)
            Thickness of the text.
        """
        img_height, img_width, _ = self.image.shape

        if categories is not None:
            colours = [color_map[category] for category in categories]
        else:
            colours = [(255, 0, 0)] * len(boxes)

        for i, (box, colour) in enumerate(zip(boxes, colours)):

            x_min, y_min, x_max, y_max = map(int, box)

            x_min = max(0, x_min - box_padding)
            y_min = max(0, y_min - box_padding)
            x_max = min(img_width, x_max + box_padding)
            y_max = min(img_height, y_max + box_padding)

            if (x_max - x_min < 1) or (y_max - y_min < 1):
                print(
                    f"Attempting to draw empty bounding box: {(x_min, y_min)} -> {(x_max, y_max)}"
                )
                continue

            self.image = cv2.rectangle(
                self.image,
                (x_min, y_min),
                (x_max, y_max),
                colour,
                thickness=line_thickness,
            )

            if texts is not None:
                _, baseline = cv2.getTextSize(
                    texts[i], cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                cv2.putText(
                    self.image,
                    texts[i],
                    (x_min, y_min - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    colour,
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )

    def draw_legend(
        self,
        origin: Tuple[int, int],
        categories: List[int],
        category_names: Dict[int, str],
        colour_map: Dict[int, Tuple[int, int, int]] = DEFAULT_COLORS,
        font_scale: float = 0.5,
        font_thickness: int = 1,
    ) -> None:
        """
        Draw a legend on the image for the specified list of categories.

        Parameters
        ----------
        origin : Tuple[int, int]
            Origin (top right corner) of the legend in pixels (X, Y)
        categories : List[int]
            List of categories for which to draw the legend.
        category_names : Dict[int, str]
            Mapping from categories to category names.
        colour_map : Dict[int, Tuple[int, int, int]] (default: DEFAULT_COLORS)
            Mapping from categories to colors.
        font_scale : float (default: 0.5)
            Font scale for the text.
        font_thickness : int (default: 1)
            Thickness of the text.
        """

        x_min, y_min = origin

        for cat in categories:
            (text_width, text_height), baseline = cv2.getTextSize(
                category_names[cat],
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )

            cv2.rectangle(
                self.image,
                (x_min - text_width, y_min - text_height - baseline),
                (x_min, y_min),
                colour_map[cat],
                thickness=cv2.FILLED,
            )
            cv2.putText(
                self.image,
                category_names[cat],
                (x_min - text_width, y_min - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA,
            )

            y_min = y_min + text_height + 2 * baseline
