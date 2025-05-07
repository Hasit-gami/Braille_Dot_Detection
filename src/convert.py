import json
import numpy as np
import torch
import pathlib


def convert_to_braille_unicode(str_input: str, path: str = "utils/braille_map.json") -> str:
    base_path = pathlib.Path(__file__).parent.resolve()
    json_path = base_path / pathlib.Path(path)
    with open(json_path, "r", encoding="utf-8") as fl:
        data = json.load(fl)

    return data.get(str_input, "")


# Braille unicode to English mapping dictionary
braille_to_english_map = {
    "⠁": "a",
    "⠃": "b",
    "⠉": "c",
    "⠙": "d",
    "⠑": "e",
    "⠋": "f",
    "⠛": "g",
    "⠓": "h",
    "⠊": "i",
    "⠚": "j",
    "⠅": "k",
    "⠇": "l",
    "⠍": "m",
    "⠝": "n",
    "⠕": "o",
    "⠏": "p",
    "⠟": "q",
    "⠗": "r",
    "⠎": "s",
    "⠞": "t",
    "⠥": "u",
    "⠧": "v",
    "⠺": "w",
    "⠭": "x",
    "⠽": "y",
    "⠵": "z",
    "⠼": "#",
    "⠴": "1",
    "⠂": "2",
    "⠆": "3",
    "⠒": "4",
    "⠲": "5",
    "⠢": "6",
    "⠖": "7",
    "⠶": "8",
    "⠦": "9",
    "⠔": "0",
    "\n": "\n",
    " ": " "
}


def translate_braille_to_english(braille_text: str) -> str:
    """
    Translate braille unicode text to English text using the braille_to_english_map dictionary.
    """
    english_text = ""
    for char in braille_text:
        english_text += braille_to_english_map.get(char, "?")  # Use '?' for unknown chars
    return english_text


def parse_xywh_and_class(boxes) -> list:
    """
    boxes input object from ultralytics YOLOv8 results
    """

    # copy values from boxes object to numpy array
    new_boxes = np.zeros((len(boxes), 6))
    new_boxes[:, :4] = boxes.xywh.numpy()  # first 4 channels are xywh
    new_boxes[:, 4] = boxes.conf.numpy()  # 5th channel is confidence
    new_boxes[:, 5] = boxes.cls.numpy()  # 6th channel is class which is last channel

    # sort according to y coordinate
    new_boxes = new_boxes[new_boxes[:, 1].argsort()]

    # find threshold index to break the line
    y_threshold = np.mean(new_boxes[:, 3]) // 2
    boxes_diff = np.diff(new_boxes[:, 1])
    threshold_index = np.where(boxes_diff > y_threshold)[0]

    # cluster according to threshold_index
    boxes_clustered = np.split(new_boxes, threshold_index + 1)
    boxes_return = []
    for cluster in boxes_clustered:
        # sort according to x coordinate
        cluster = cluster[cluster[:, 0].argsort()]
        boxes_return.append(cluster)

    return boxes_return
