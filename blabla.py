import csv



def load_csv_file(filename, skip_columns=0):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
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

def remove_1s(results):
    for i, k in enumerate(results.keys()):
        segs = results[k]
        if len(segs) > 1:
            results[k] = results[k][1:]
    return results

def write_csv(filename, results):
    csv_res = [['ImageId', 'EncodedPixels']]
    print("write csv")
    for k in results.keys():
        segs = results[k]
        for seg in segs:
            csv_res.append([k, seg])
    print(len(csv_res))
    print("done with appending segs")
    # Write CSV file
    with open(filename+'.csv', 'w') as fp:
        writer = csv.writer(fp) # , encoding='utf-8')
        # writer.writerow(["your", "header", "foo"])  # write header
        writer.writerows(csv_res)
    print("Wrote file: "+ filename+".csv")

if __name__ == '__main__':
    results = load_csv_file(filename="output/test2.csv")
    results = remove_1s(results)
    write_csv("without_1s.csv", results)