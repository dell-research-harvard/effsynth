import os
from matplotlib.pyplot import text
from tqdm import tqdm
import json
import argparse

from utils.fonts import load_chars, get_unicode_coverage_from_ttf
from utils.coco import create_coco_annotation_field, COCO_JSON_SKELETON
from core.core import TextlineGenerator
from utils.transforms import TRANSFORM_DICT


if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True, default=1_000,
        help="Number of textlines to generate")
    parser.add_argument("--language", type=str, required=True, default="en",
        help="Language of text being generated")
    parser.add_argument("--font_folder", type=str, required=True, default="./fonts/en",
        help="Path to folder with font files of interest")
    parser.add_argument("--char_folder", type=str, required=True, default="./chars/en",
        help="Path to folder with character set text files of interest")
    parser.add_argument("--char_sets", type=str, required=True, default="latin,numerals,punc_basic",
        help="Names of character sets to utilize as a comma separated list")
    parser.add_argument("--char_set_props", type=str, required=True, default="1.0,0.0,0.0",
        help="Proportions of character sets to use for generation as a comma separated list")
    parser.add_argument("--train_test_val_props", type=str, default="0.8,0.2,0.0",
        help="Proportions of train/test/val sets as a comma separated list")
    parser.add_argument("--output_folder", type=str, default="./output",
        help="Path to folder for saving output")
    parser.add_argument("--textline_max_numbers", type=int, default=2,
        help="The maximum number of numbers to generate in a textline")
    parser.add_argument("--textline_numbers_geom_p", type=float, default=0.005,
        help="The p parameter in a geometric distribution, used for sampling numbers")
    parser.add_argument("--font_sizes", type=str, default="64",
        help="The size of a textline's font as a comma separated list")
    parser.add_argument("--textline_max_length", type=int, default=20,
        help="The max number of characters in a textline")
    parser.add_argument("--textline_max_spaces", type=int, default=5,
        help="The max number of spaces in a textline")
    parser.add_argument('--transforms',
        choices=['default', 'pr', 'album', 'trdg', 'trdgcolor'], type=str, default="default",
        help="Option for transforming synthetically rendered textline")
    parser.add_argument('--vertical', action='store_true', default=False,
        help="Generate vertically oriented textlines")
    parser.add_argument("--char_dist", type=int, default=0,
        help="Distance between characters in pixels")
    parser.add_argument("--char_dist_std", type=int, default=2,
        help="Distance between characters in pixels")
    parser.add_argument("--specific_seqs", type=str, default=None,
        help="Assert specific character sequences appear in renders at random, separated by pipes")
    parser.add_argument("--p_spec_seqs", type=str, default=None,
        help="The probability each specific sequence is included in a generated textline")
    parser.add_argument("--spec_seq_count", type=int, default=1,
        help="The number of specific sequences to be inserted")
    parser.add_argument('--word_bbox', action='store_true', default=False,
        help="Create bboxes for words too")
    parser.add_argument('--single_words', action='store_true', default=False,
        help="Generate images of single words or random character strings")
    parser.add_argument('--real_words', type=int, default=0,
        help="Number of real words to insert in generated textlines")
    parser.add_argument('--wiki_text', action='store_true', default=False,
        help="Pull generated text sequences randomly from Wikipedia")
    args = parser.parse_args()

    # create transforms
    synth_transform = TRANSFORM_DICT[args.transforms]
    
    # get font paths
    font_paths = [os.path.join(args.font_folder, x) for x in os.listdir(args.font_folder)]

    # make coverage dict
    coverage_dict = {}
    for font_path in font_paths:
        _, covered_chars = get_unicode_coverage_from_ttf(font_path)
        coverage_dict[font_path] = covered_chars

    # get char paths
    char_paths = [os.path.join(args.char_folder, x) for x in os.listdir(args.char_folder)]
    char_sets = args.char_sets.split(",")
    chosen_char_paths = [[x for x in char_paths if c in x][0] for c in char_sets]
    print(f"Chosen character sets: {chosen_char_paths}")
    char_set_props = [float(x) for x in args.char_set_props.split(",")]
    assert 0.9999999 < sum(char_set_props) <= 1, f"Character set proportions do not sum to 1! They sum to {sum(char_set_props)}!"
    char_set_lists = [load_chars(x) for x in chosen_char_paths]
    char_sets_and_props = list(zip(char_set_lists, char_set_props))
    
    # create output folder
    outdir = args.output_folder
    os.makedirs(outdir, exist_ok=True)

    # train test val split
    train_test_val_split = [float(x) for x in args.train_test_val_props.split(",")]
    train_test_val_counts = [int(args.count * x) for x in train_test_val_split]
    
    # set dicts
    SETNAMES = ("train", "test", "val",)
    anns_dict = {SETNAMES[0]: [], SETNAMES[1]: [], SETNAMES[2]: []}
    images_dict = {SETNAMES[0]: [], SETNAMES[1]: [], SETNAMES[2]: []}
    anno_id = 0

    # save for images
    images_path = os.path.join(outdir, "images")
    os.makedirs(images_path, exist_ok=True)

    # create segs
    for setname, count in zip(SETNAMES, train_test_val_counts):

        textline_generator = TextlineGenerator(
            setname, font_paths, char_sets_and_props, images_path, 
            synth_transform, coverage_dict,
            args.textline_max_length, args.font_sizes, args.textline_max_spaces,
            args.textline_numbers_geom_p, args.textline_max_numbers,
            args.language, args.vertical, args.specific_seqs,
            args.char_dist, args.char_dist_std, args.p_spec_seqs,
            args.word_bbox, args.real_words, args.single_words,
            args.spec_seq_count, args.wiki_text
        )

        for image_id in tqdm(range(count)):

            textline_dict = textline_generator.generate_synthetic_textline(image_id=image_id)

            synth_text = textline_dict["text"]
            synth_image = textline_dict["trans_image"]
            image_name = textline_dict["image_name"]

            if all(c == "_" for c in synth_text):
                continue

            imgw, imgh = synth_image.width, synth_image.height
            image = {"width": imgw, "height": imgh, "id": image_id, 
                "file_name": image_name, "text": synth_text.replace("_", " ")}
            images_dict[setname].append(image)

            for bbox in textline_dict.get("bboxes", list()):
                x, y, width, height = bbox
                x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x+width, imgw), min(y+height, imgh)
                x, y, width, height = x0, y0, x1 - x0, y1 - y0
                annotation = create_coco_annotation_field(anno_id, image_id, width, height, x, y, cat_id=0)
                anns_dict[setname].append(annotation)
                anno_id += 1

            for bbox in textline_dict.get("word_bboxes", list()):
                x, y, width, height = bbox
                x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x+width, imgw), min(y+height, imgh)
                x, y, width, height = x0, y0, x1 - x0, y1 - y0
                annotation = create_coco_annotation_field(anno_id, image_id, width, height, x, y, cat_id=1)
                anns_dict[setname].append(annotation)
                anno_id += 1

    # output
    for setname, pct in zip(SETNAMES, train_test_val_split):
        coco_json = COCO_JSON_SKELETON.copy()
        coco_json["images"] = images_dict[setname]
        coco_json["annotations"] = anns_dict[setname]
        with open(os.path.join(outdir, f"{setname}{int(pct*100)}.json"), 'w') as f:
            json.dump(coco_json, f, indent=2)

    # charset
    with open(os.path.join(outdir, f"charset.txt"), 'w') as f:
        f.write("\n".join(str(ord(c)) for c in list(textline_generator.all_chars)))
