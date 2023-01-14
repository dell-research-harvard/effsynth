

COCO_JSON_SKELETON = {
        "images": [],
        "annotations": [],
        "info": {"year": 2022, "version": "1.0", "contributor": "synth-textlines"},
        "categories": [{"id": 0, "name": "character"}, {"id": 1, "name": "word"}],
        "licenses": ""
    }


def create_coco_annotation_field(anno_id, image_id, width, height, x, y, cat_id):
    return {
        "id": anno_id, 
        "image_id": image_id, 
        "category_id": cat_id,
        "area": int(width*height), 
        "bbox": [int(x), int(y), int(width), int(height)],
        "segmentation": [[int(x), int(y), int(x)+int(width), int(y), 
            int(x)+int(width), int(y)+int(height), int(x), int(y)+int(height)]],
        "iscrowd": 0,
        "ignore": 0
    }


def create_coco_anno_entry(x, y, w, h, ann_id, image_id):
    return {
        "segmentation": [[int(x), int(y), int(x)+int(w), int(y), 
                          int(x)+int(w), int(y)+int(h), int(x), int(y)+int(h)]], 
        "area": int(w)*int(h), "iscrowd": 0, 
        "image_id": image_id, "bbox": [int(x), int(y), int(w), int(h)], 
        "category_id": 0, "id": ann_id, "score": 1.0
    }


def create_coco_image_entry(path, h, w, image_id):
    return {
        "file_name": path, 
        "height": h, 
        "width": w, 
        "id": image_id
    }