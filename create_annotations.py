#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

dataset = "test"

ROOT_DIR = 'detectron/datasets/data/dsb18'
IMAGE_DIR = os.path.join(ROOT_DIR, dataset)
#ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")
ANNOTATION_DIR = "/home/johan/PycharmProjects/convert_data/dsb18/annotations"#train/annotations"


INFO = {
    "description": "dsb18",
    "url": "https://github.com/jolohan",
    "version": "0",
    "year": 2018,
    "contributor": "jolohan",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "None",
        "url": "https://kaggle.com"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'none',
        'supercategory': 'none',
    },
    {
        'id': 2,
        'name': 'none2',
        'supercategory': 'none',
    },
]

# also  png files
def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png'] # also  png files
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    #print(IMAGE_DIR)
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
        for image_filename in image_files:
            print(image_filename)
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    #print(annotation_filename)
                    if 'none' in annotation_filename:
                        class_id = 1
                    elif 'circle' in annotation_filename:
                        class_id = 2
                    else:
                        class_id = 3

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open(('{}/' + dataset + '.json').format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
        print(output_json_file)


if __name__ == "__main__":
    main()