import json
from pprint import pprint
import numpy as np
from detectron.datasets.nuclei.mask_encoding import rle_encode
from pycocotools import mask as mask_util
import unicodecsv as csv
from scipy import ndimage
import sys
import time
from datetime import datetime, timedelta



def join_results(seg_filename, annotation_filename, result_filename,
                 intersection_thresh=0.3, mask_area_threshold=15,
                 accuracy_threshold=0.5):
    TOL = 0.00001
    annotations = load_json_file(annotation_filename)
    hex_to_id, id_to_hex = make_hex_to_id_dic(annotations['images'])
    #print(annotations['images'][0])
    segmentations = load_json_file(seg_filename)
    #print(len(segmentations))
    #print(segmentations[0])

    csv_res = [['ImageId', 'EncodedPixels']]

    t0 = time.time()
    t1 = time.time()
    old_im_id = None
    all_masks = None
    all_masks_no_refine = None
    img_ids = []

    segmentations_per_image = {}
    for i, seg in enumerate(segmentations):
        im_id = seg['image_id']
        if not segmentations_per_image.has_key(im_id):
            segmentations_per_image[im_id] = []
        segmentations_per_image[im_id].append(seg)

    """for i, id_key in enumerate(id_to_hex.keys()):
        im_name = id_to_hex[id_key]
        if id_key in segmentations_per_image:
            pass
            #csv_res.append([im_name, "1 1"])
        else:
            csv_res.append([im_name, "1 1"])"""
    n = len(segmentations_per_image.keys())
    for i, k in enumerate(segmentations_per_image.keys()):
        all_segs_in_image = segmentations_per_image[k]
        if i % 10 == 0:
            t2 = time.time()
            print(i/(n*1.0))
            print("Processing file {} ({}%)".format(i, 100 * i // n)) #, end="")
            print(" {}s (total: {}s)".format(t2 - t1, t2 - t0))
            time_left = ((t2 - t0) / (i + TOL)) * (n-i)
            time_left = ((t2 - t0)/(1.0*(i+TOL))) * (n-i)
            #print("Time left:")
            #print_time(time_left)
            #print("\n")
            t1 = t2



        all_masks = None
        all_masks_no_refine = None

        for mask_number, mask in enumerate(all_segs_in_image):
            #print(mask)
            im_id = mask['image_id']
            rle = mask['segmentation']
            score = mask['score']
            if score < accuracy_threshold:
                continue


            mask_int_orig = mask_util.decode(rle)
            #u_rle = rle_encode(mask_int)
            #print("u_rle:", u_rle)

            mask_int = ndimage.morphology.binary_fill_holes(mask_int_orig.copy()).astype(np.uint8)
            mask = mask_int > 0
            mask_orig = mask_int_orig > 0

            if all_masks is None:
                all_masks = mask.copy()
                all_masks[:] = False
                all_masks_no_refine = mask_orig.copy()
                all_masks_no_refine[:] = False

            intersection = mask & all_masks

            area_inter = intersection.sum()
            if area_inter > 0:
                # print("Area intersection > 0")
                total_area = mask.sum()
                if float(area_inter) / (float(total_area) + TOL) > intersection_thresh:
                    continue

            mask = mask & ~all_masks
            if mask.sum() < mask_area_threshold:
                continue

            mask_int[~mask] = 0

            # add this to all_masks mask
            all_masks = mask | all_masks
            all_masks_no_refine = all_masks_no_refine | mask_orig

            # rle = mask_util.encode(np.asarray(mask_int, order='F'))
            # u_rle = mask_util.decode(rle)
            u_rle = np.asarray(mask_int, order='F')

            u_rle = rle_encode(u_rle)

            # this
            # u_rle = mask_util.decode(rle)
            # u_rle = rle_encode(u_rle)
            # this

            #print(u_rle)
            im_name = id_to_hex[im_id]
            #print(u_rle)
            #print(u_rle[0])
            u_rle = [''.join(str(x)) for x in zip(u_rle[0::2], u_rle[1::2])]
            u_rle = [x.replace(",", "") for x in u_rle]
            u_rle = [x[1:-1] for x in u_rle]
            u_rle = " ".join(x for x in u_rle)
            #if (i % 1000 == 0):
            #    print(u_rle)
            if u_rle.strip():
                csv_res.append([im_name, u_rle])

    write_csv(result_filename, csv_res)

def write_csv(filename, data):
    # Write CSV file
    with open(filename+'.csv', 'w') as fp:
        writer = csv.writer(fp, encoding='utf-8')
        # writer.writerow(["your", "header", "foo"])  # write header
        writer.writerows(data)
    print("Wrote file: "+ filename)

def print_time(seconds):
    sec = timedelta(seconds)
    d = datetime(1, 1, 1) + sec
    print("MIN:SEC")
    print("%d:%d" % (d.minute, d.second))




def load_json_file(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def make_hex_to_id_dic(annotations):
    hex_to_id = {}
    id_to_hex = {}
    #print(len(annotations))
    for i, image in enumerate(annotations):
        #print(image)
        #print(image['file_name'])
        filename = image['file_name'].split('.')[0]
        image_id = image['id']
        hex_to_id[filename] = image_id
        id_to_hex[image_id] = filename
        #print(image_id, filename)
        #print(hex_to_id["jjgj"]) # make it crash
    return hex_to_id, id_to_hex

def write_results(seg_filename, annotation_filename, result_filename):

    join_results(seg_filename=seg_filename,
                 annotation_filename=annotation_filename,
                 result_filename=result_filename)

if __name__ == '__main__':
    #annotation_filename = 'detectron/datasets/data/dsb18/annotations/test'
    # seg_filename = "/detectron/output/test/dsb18_test/generalized_rcnn/" \
    #              "segmentations_dsb18_test_results.json"
    seg_filename = "segm_18000.json"
    annotation_filename = 'detectron/datasets/data/dsb18/annotations/stage1_test.json'
    result_filename = 'test2.csv'
    if len(sys.argv) > 1:
        seg_filename = sys.argv[1]
    if len(sys.argv) > 2:
        annotation_filename = sys.argv[2]
    if len(sys.argv) > 3:
        result_filename = sys.argv[3]

    print("Writing results from seg file: ", seg_filename)
    print("And annotations file: ", annotation_filename)
    #seg_filename = "segmentations_dsb18_test_results"
    write_results(seg_filename=seg_filename,
                  annotation_filename=annotation_filename,
                  result_filename=result_filename)