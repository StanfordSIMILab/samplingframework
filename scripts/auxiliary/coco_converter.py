import os
import json
import numpy as np

def convert_to_coco(self, image_path, annotation_path, categories=None):
    if categories is None:
        # Default categories are derived from our Microvascular Decompression dataset,
        categories = [
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
    else:
        categories = categories

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
                data['category_id'] = coco_categories[label]["id"]
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

def convert_selected_frames_to_coco(self, selected_frames, output_file_path, categories=None):
    if categories is None:
            # Default categories are derived from our Microvascular Decompression dataset,
            # with IDs and names based on folder names in the original data structure
            categories = [
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
        else:
            categories = categories

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
                data['category_id'] = self.categories[label]["id"]

                data['tags'] =  [str([self.categories[label]["id"]])]
                
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
                "metadata": self.categories[label]["id"],  # Include full metadata if needed
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

def convert_coco_to_png_masks(
    coco_annotations: Dict,
    images_dir: str,
) -> None:
    coco_data = coco_annotations

    category_map = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
    image_info = {img["id"]: img for img in coco_data["images"]}

    frame_annots: Dict[str, list] = {}
    for ann in coco_data["annotations"]:
        frame_id = ann["image_id"]
        frame_annots.setdefault(frame_id, []).append(ann)

    for image_id, anns in tqdm(frame_annots.items(), desc="Creating masks"):
        img_info = image_info.get(image_id)
        if img_info is None:
            continue

        h, w = img_info["height"], img_info["width"]
        frame_name = os.path.splitext(os.path.basename(img_info["file_name"]))[0]

        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for ann in anns:
            category_id = ann.get("category_id", 0)
            seg = ann.get("segmentation")

            if seg is None:
                print(f"  Warning: ann {ann.get('id')} on frame {image_id} has no segmentation — skipping")
                continue

            if isinstance(seg, list):
                polys = seg if (len(seg) > 0 and isinstance(seg[0], list)) else [seg]
                for poly in polys:
                    if len(poly) < 6:
                        print(
                            f"  Warning: ann {ann.get('id')} on frame {image_id} has degenerate "
                            f"polygon with {len(poly)} values (need ≥ 6) — skipping polygon"
                        )
                        continue
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(combined_mask, [pts], category_id)

            elif isinstance(seg, dict):
                if mask_utils is None:
                    print(f"  Warning: skipping RLE mask for ann {ann['id']} — pycocotools not installed")
                    continue
                try:
                    if isinstance(seg.get("counts"), list):
                        rle = mask_utils.frPyObjects(seg, seg["size"][0], seg["size"][1])
                    else:
                        rle = seg
                    rle_mask = mask_utils.decode(rle)
                    combined_mask[rle_mask > 0] = category_id
                except Exception as e:
                    print(f"  Warning: failed to decode RLE mask for ann {ann.get('id')} on frame {image_id}: {e} — skipping")
                    continue

        if combined_mask.max() == 0:
            print(f"  Warning: frame {image_id} produced an empty mask — skipping")
            continue

        out_path = os.path.join(images_dir, f"{frame_name}_mask.png")
        Image.fromarray(combined_mask).save(out_path)

    print(f"  ✓ Masks saved under {images_dir}")

    