import os
import json
import numpy as np

categories =  [
    {"id": 1, "name": "Cerebellum", "supercategory": "shape"},
    {"id": 2, "name": "Arachnoid", "supercategory": "shape"},
    {"id": 3, "name": "CN8", "supercategory": "shape"},
    {"id": 4, "name": "CN5", "supercategory": "shape"},
    {"id": 5, "name": "CN7", "supercategory": "shape"},
    {"id": 6, "name": "CN_9_10_11", "supercategory": "shape"},
    {"id": 7, "name": "SCA", "supercategory": "shape"},
    {"id": 8, "name": "AICA", "supercategory": "shape"},
    {"id": 9, "name": "SuperiorPetrosalVein", "supercategory": "shape"},
    {"id": 10, "name": "Labrynthine", "supercategory": "shape"},
    {"id": 11, "name": "Vein", "supercategory": "shape"},
    {"id": 12, "name": "Brainstem", "supercategory": "shape"},
    {"id": 1001, "name": "Suction", "supercategory": "shape"},
    {"id": 1002, "name": "Bovie", "supercategory": "shape"},
    {"id": 1003, "name": "Bipolar", "supercategory": "shape"},
    {"id": 1004, "name": "Forcep", "supercategory": "shape"},
    {"id": 1005, "name": "BluntProbe", "supercategory": "shape"},
    {"id": 1006, "name": "Drill", "supercategory": "shape"},
    {"id": 1007, "name": "Kerrison", "supercategory": "shape"},
    {"id": 1008, "name": "Cottonoid", "supercategory": "shape"},
    {"id": 1009, "name": "Scissors", "supercategory": "shape"},
    {"id": 1012, "name": "Unknown", "supercategory": "shape"},
    {"id": 1023, "name": "Dissector", "supercategory": ""},
    {"id": 1024, "name": "Teflon", "supercategory": ""}
]

def convert_to_coco(image_path, annotation_path):

        coco_annotations = []
        coco_images = []
        c = 0

        for filename in os.listdir(annotation_path):

            framename = filename.split('.')[0]
            frame_num = framename.split('.')[0][-4:]

            json_file_path = os.path.join(annotation_path,framename + '.json')
            with open(json_file_path, 'r') as f:
                result = json.load(f)

            if 'labels' in result and 'bboxes' in result and 'scores' in result and 'masks' in result:
                labels = result['labels']
                bboxes = result['bboxes']
                scores = result['scores']

                # segm results
                masks = result['masks']
                for i, label in enumerate(labels):
                    data = dict()
                    data['image_id'] = int(frame_num)

                    x1 = bboxes[i][0]
                    y1 = bboxes[i][1]
                    x2 = bboxes[i][2]
                    y2 = bboxes[i][3]
                    coco_bbox = [x1, y1, x2-x1, y2-y1]
                    data['bbox'] = list(int(np.round(x)) for x in coco_bbox)
                    data['score'] = float(scores[i])
                    if data['score'] <= 0.7:
                        continue
                    data['category_id'] = categories[label]["id"]
                    if isinstance(masks[i]['counts'], bytes):
                        masks[i]['counts'] = masks[i]['counts'].decode()
                    data['segmentation'] = masks[i]
                    data['id'] = c + 1
                    c = c + 1
                    data["iscrowd"] = 0
                    coco_annotations.append(data)

                coco_image = {
                    "id": int(frame_num),
                    "width": 1920,
                    "height": 1080,
                    "file_name": framename + '.jpg',
                }

                coco_images.append(coco_image)
            else:
                print(f"Missing keys in result: {result.keys()}")

        return coco_images, coco_annotations

def convert_selected_frames_to_coco(selected_frames, output_file_path):
    coco_annotations = []
    coco_images = []
    coco_categories = {}
    c = 0  # Counter for unique annotation IDs

    for frame_path in selected_frames:
        # Normalize path for cross-platform compatibility
        frame_path = frame_path.replace("\\", "/")
        coco_categories = {category["name"]: category for category in categories}

        # Extract frame information
        framename = os.path.basename(frame_path).split('.')[0]  # Extract file name without extension
        frame_num = framename.split('_')[-1]  
        with open(frame_path, 'r') as f:
            result = json.load(f)

        # Check if required keys exist in the JSON file
        if 'labels' in result and 'bboxes' in result and 'scores' in result and 'masks' in result:
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            masks = result['masks']

            file_path = frame_path.replace('\\', '/')
            file_path = file_path.replace('/data/ground_truth', '/output/video_frames')
            file_path = file_path.replace('/preds', '')
            file_path = file_path.replace('.json', '.jpg')
            file_path = file_path.replace("\\", "/")
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = file_path#int(frame_num)

                # Convert bounding box to COCO format: [x_min, y_min, width, height]
                x1 = bboxes[i][0]
                y1 = bboxes[i][1]
                x2 = bboxes[i][2]
                y2 = bboxes[i][3]
                coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                data['bbox'] = list(int(np.round(x)) for x in coco_bbox)
                
                # Add score and filter by threshold
                data['score'] = float(scores[i])
                if data['score'] <= 0.7:
                    continue

                # Map category ID using the folder name
                data['category_id'] = categories[label]["id"]

                data['tags'] =  [str([categories[label]["id"]])]
                
                # Handle segmentation masks
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]

                data['id'] = c + 1  # Unique annotation ID
                c += 1
                data["iscrowd"] = 0

                coco_annotations.append(data)

            # Add image metadata to coco_images
            coco_image = {
                "id": file_path,#int(frame_num),
                "width": 1920,  # Assuming fixed dimensions; adjust if necessary
                "height": 1080,
                "file_name": file_path,#framename + '.jpg',
                "metadata": categories[label]["id"],  # Include full metadata if needed
            }
            coco_images.append(coco_image)

    # Create COCO dictionary structure
    coco_output = {
        "info": {
            "description": "COCO dataset generated from selected frames",
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": list(coco_categories.values()),  # Convert categories dict to list
    }

    # Save the output to the specified JSON file
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as f:
        json.dump(coco_output, f)

    print(f"COCO file created at: {output_file_path}")