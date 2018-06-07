
import csv
import numpy as np
import os.path
import sys

def load_csv_file(filename, skip_columns=0):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        segmentations_per_image = {}
        for row in reader:
            image_name = row[0]
            segm = row[1:]
            if skip_columns > 0:
                segm = row[1:(-1*skip_columns)]

            if not segmentations_per_image.has_key(image_name):
                segmentations_per_image[image_name] = []

            segmentations_per_image[image_name].append(segm)
        return segmentations_per_image

def make_all_segs_all_pixels(all_segs):
    all_segs_all_pixels = [[]] * len(all_segs)
    for i, seg in enumerate(all_segs):
        all_pixels = get_all_pixels_from_rle(seg[0])
        all_segs_all_pixels[i] = all_pixels
    return all_segs_all_pixels

def get_score(results, ground_truth):
    if len(results.keys()) != len(ground_truth.keys()):
        print("Length does not match.")
        print("Result images length: ", len(results.keys()))
        print("Ground truth images length: ", len(ground_truth.keys()))
    all_image_scores = []
    for image_name in ground_truth.keys():
        image_scores_at_threshold = []

        ground_truth_segs = ground_truth[image_name]
        ground_truth_segs = make_all_segs_all_pixels(ground_truth_segs)

        if image_name in results:
            results_segs = results[image_name]
            results_segs = make_all_segs_all_pixels(results_segs)

        else:
            print("No results for image: ", image_name)
            image_scores_at_threshold.append(0)
            continue

        best_match = [0.0]*len(results_segs)
        #print(best_match)
        for i, results_seg in enumerate(results_segs):
            for j, ground_truth_seg in enumerate(ground_truth_segs):
                overlapping_score = get_overlapping_score(results_seg, ground_truth_seg)
                if overlapping_score > best_match[i]:
                    best_match[i] = overlapping_score
        #print(best_match)

        number_of_hits_all_thresholds = 0
        for x, score in enumerate(best_match):
            hits = int((score - 0.45)*100) / 5
            hits = max(hits, 0)
            number_of_hits_all_thresholds += hits
            #print(score, hits)

        number_of_misses_all_thresholds = len(results_segs)*10 - number_of_hits_all_thresholds
        all_positives_all_thresholds = len(ground_truth_segs)*10
        false_negatives_all_thresholds = all_positives_all_thresholds - number_of_hits_all_thresholds

        #print(number_of_hits_all_thresholds)
        #print(number_of_misses_all_thresholds)
        #print(false_negatives_all_thresholds)
        image_score = number_of_hits_all_thresholds*1.0 / (number_of_hits_all_thresholds +
                                                        number_of_misses_all_thresholds +
                                                        false_negatives_all_thresholds)

        #print(image_score)
        all_image_scores.append(image_score)
        print(np.average(all_image_scores))

    return np.average(all_image_scores)


def get_overlapping_score(seg1, seg2):
    #if (seg1[0] - seg2[0] > 10000):
    #    return -1
    #print(all_pixels_1)
    #print(all_pixels_2)
    intersection = np.intersect1d(seg1, seg2)
    union = np.union1d(seg1, seg2)

    return len(intersection)*1.0 / len(union)

def get_all_pixels_from_rle(rle):
    all_pixels = []
    rle_list =  rle.split(" ")
    for i in range(0, len(rle_list)-1, 2):
        pixel_start = int(rle_list[i])
        pixel_run = int(rle_list[i+1])
        pixel_end = pixel_start + pixel_run
        all_pixels += [x for x in range(pixel_start, pixel_end)]

    return all_pixels

def append_score_to_csv_file(score, filename, model):
    if os.path.isfile(filename):
        with open (filename, "a") as f:
            f.write('\n' + model + "," + str(score))
    else:
        with open(filename, 'w') as f:
            f.write(model + "," + str(score))

def evaluate(filename_results, filename_solution, filename_write_score, model):
    print("Evaluating")
    results = load_csv_file(filename=filename_results)
    ground_truth = load_csv_file(filename=filename_solution, skip_columns=3)
    score = get_score(results=results, ground_truth=ground_truth)
    append_score_to_csv_file(score=score, filename=filename_write_score, model=model)

if __name__ == '__main__':
    filename_results = "stage1_test.csv"
    model = "standard"
    config = 'e2e_mask_rcnn_X-101-64x4d-FPN_1x_v2'
    if len(sys.argv) > 3:
        results_root = sys.argv[1]
        config = sys.argv[2]
        model = sys.argv[3]
        filename_results = results_root + config + '___' + model + ".csv"
    else:
        print("evaluate takes 3 arguments")
    evaluate(filename_results=filename_results,
             filename_solution="detectron/datasets/data/dsb18/stage1_solution.csv",
             filename_write_score='output/all_scores_' + config + '.csv',
             model=model)
    #output/180000_iter___e2e_mask_rcnn_R-101-FPN_1x/