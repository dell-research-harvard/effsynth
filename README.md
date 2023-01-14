# EffSynth

## Introduction

This repository can be used to re-produce the synthetic OCR training datasets used for pre-training EffOCR and other models that are compared to EffOCR. The only script necessary to create synthetic datasets is `effsynth.py`. For documentation on arguments that can/need to be supplied to this script, simply run

```
python effsynth.py -h
```

All dataset annotations are exported in a COCO format.

## Examples

Otherwise, please see the following examples for creating synthetic OCR training datasets with EffSynth.

English random character sequence dataset example:
```
python effsynth.py --count 10000 --language en --font_folder fonts/en --char_folder chars/en --char_sets latin,numeral,punc --char_set_props 0.7,0.1,0.2 --train_test_val_props 0.8,0.1,0.1 --output_folder /path/to/output/dir --textline_max_numbers 2 --font_sizes 64 --textline_max_length 5 --textline_max_spaces 3 --transforms trdgcolor --char_dist 0 --char_dist_std 2 --specific_seqs ",|.|-" --p_spec_seqs 0.4,0.4,0.2 --spec_seq_count 2 --word_bbox --real_words 3
```

Vertical Japanese random character sequence dataset example:
```
python effsynth.py --count 10000 --language jp --font_folder fonts/jp --char_folder chars/jp --char_sets adobe,jis,hiragana,katakana,numeral,punc --char_set_props 0.35,0.35,0.1,0.1,0.05,0.05 --train_test_val_props 0.8,0.1,0.1 --output_folder /path/to/output/dir --textline_max_numbers 2 --font_sizes 64 --textline_max_length 10 --textline_max_spaces 2 --transforms trdgcolor --char_dist 15 --char_dist_std 2 --vertical
```

English random text sequence dataset example:
```
python effsynth.py --count 10000 --language en --font_folder fonts/en --char_folder chars/en --char_sets latin,numeral,punc --char_set_props 0.7,0.1,0.2 --train_test_val_props 0.8,0.1,0.1 --output_folder /path/to/output/dir --textline_max_numbers 2 --font_sizes 64 --textline_max_length 20 --textline_max_spaces 3 --transforms trdgcolor --char_dist 0 --char_dist_std 2 --specific_seqs ",|.|-" --p_spec_seqs 0.4,0.4,0.2 --spec_seq_count 2 --word_bbox --real_words 3 --wiki_text
```
