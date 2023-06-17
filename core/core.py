import numpy as np
from PIL import ImageOps, Image, ImageFont, ImageDraw
import os
import re
import wikipedia

from utils.misc import *


class TextlineGenerator:

    def __init__(
            self, setname, font_paths, char_sets_and_props, save_path, 
            synth_transform, coverage_dict,
            max_length, font_sizes, max_spaces, num_geom_p, max_numbers,
            language, vertical, spec_seqs, char_dist, char_dist_std,
            p_specseq, word_bbox, real_words, single_words, specseq_count,
            wiki_text
        ):

        self.setname = setname
        self.font_paths = font_paths
        self.char_sets_and_props = char_sets_and_props
        self.char_sets = [char_set for char_set, prop in self.char_sets_and_props]
        self.all_chars = set(sum(self.char_sets, []))
        self.save_path = save_path
        self.synth_transform = synth_transform
        self.coverage_dict = coverage_dict
        self.max_length = max_length
        self.font_sizes = [int(x) for x in font_sizes.split(",")]
        self.max_spaces = max_spaces
        self.num_geom_p = num_geom_p
        self.max_numbers = max_numbers
        self.low_chars = ",.ygjqp" if language == "en" else ""
        self.language = language
        self.vertical = vertical
        self.spec_seqs = spec_seqs.split("|") if not spec_seqs is None else None
        self.specseq_count = specseq_count 
        self.char_dist = char_dist
        self.char_dist_std = char_dist_std
        self.p_specseq = [float(x) for x in p_specseq.split(",")] if not p_specseq is None else None
        if not self.p_specseq is None:
            assert len(self.p_specseq) == len(self.spec_seqs)
            assert round(sum(self.p_specseq), 4) == 1., f"Probs of spec seqs do not add to 1! ({sum(self.p_specseq)})"
        self.word_bbox = word_bbox
        if real_words > 0 or single_words:
            assert os.name == "posix", "Not a unix OS; adding in real words won't work!"
            with open("/usr/share/dict/words", "r") as f:
                words = re.sub("[^\w]", " ",  f.read()).split()
            self.words = words
        self.num_real_words = real_words
        self.single_words = single_words
        self.wiki_text = wiki_text

    def select_font(self):

        font_path = np.random.choice(self.font_paths)
        self.font_size = int(np.random.choice(self.font_sizes))
        self.digital_font = ImageFont.truetype(font_path, size=self.font_size)
        self.covered_chars = set(self.coverage_dict[font_path])

    def generate_synthetic_textline_text(self):

        seq_chars = []
        num_chars = np.random.choice(range(1, self.max_length))
        props = [x[1] for x in self.char_sets_and_props]
        for char_set, prop in self.char_sets_and_props:
            if prop == max(props) and round(prop * num_chars) == 0:
                char_set_count = 1
            else:
                char_set_count = round(prop * num_chars)
            available_chars = self.covered_chars.intersection(set(char_set))
            chosen_chars = np.random.choice(list(available_chars), char_set_count)
            seq_chars.extend(chosen_chars)
        
        num_spaces = np.random.choice(range(0, self.max_spaces))
        seq_spaces = num_spaces * ["_"]
        
        num_numbers = np.random.choice(range(0, self.max_numbers))
        seq_numbers = np.random.geometric(p=self.num_geom_p, size=num_numbers)

        synth_seq = to_string_list(seq_numbers) + to_string_list(seq_spaces) + to_string_list(seq_chars)

        if not self.spec_seqs is None:
            for _ in range(self.specseq_count):
                seq_spec = np.random.choice(self.spec_seqs, p=self.p_specseq)
                synth_seq += [seq_spec]

        if self.num_real_words > 0:
            random_words = np.random.choice(self.words, self.num_real_words).tolist()
            synth_seq += [f"_{w}_" for w in random_words]
            synth_seq

        np.random.shuffle(synth_seq)
        synth_text = "".join(synth_seq)
        synth_text = re.sub("_+",  "_", synth_text)
        self.num_symbols = len(synth_text)

        assert len(synth_text) > 0, synth_text

        return synth_text

    def generate_synthetic_word_text(self):

        random_word = np.random.choice(self.words)

        random_chars = []
        num_chars = np.random.choice(range(1, self.max_length))
        for char_set, prop in self.char_sets_and_props:
            char_set_count = round(prop * num_chars)
            available_chars = self.covered_chars.intersection(set(char_set))
            chosen_chars = np.random.choice(list(available_chars), char_set_count)
            random_chars.extend(chosen_chars)
        np.random.shuffle(random_chars)
        random_chars = "".join(random_chars)

        if not self.spec_seqs is None:
            seq_spec = np.random.choice(self.spec_seqs, p=self.p_specseq)

        synth_text = np.random.choice([
            random_chars + seq_spec, 
            seq_spec + random_chars, 
            random_word + seq_spec, 
            seq_spec + random_word
        ])

        self.num_symbols = len(synth_text)

        return synth_text

    def generate_synthetic_wiki_text(self):

        wikipedia.set_lang(self.language)
        wikipedia.set_rate_limiting(rate_limit=True)

        random_page = None
        while random_page is None:
            random_page = self.wiki_check("https://en.wikipedia.org/wiki/Special:Random")
        
        random_content = self.clean_wiki_text(random_page.content)

        num_chars = np.random.choice(range(1, self.max_length))
        synth_text = " "

        while str.isspace(synth_text):
            random_start_idx = np.random.choice(range(0, len(random_content) - num_chars))
            synth_text = random_content[random_start_idx:random_start_idx+num_chars]
            
        self.num_symbols = len(synth_text)

        return synth_text
        

    def generate_synthetic_textline_image_latin_based(self, text):

        W = sum(self.digital_font.getsize(c)[0] + self.char_dist for c in text) - (2 * self.char_dist)
        H = self.digital_font.getsize(text)[1]
        image = Image.new("RGB", (W, H), (255,255,255))
        draw = ImageDraw.Draw(image)
        x_pos, y_pos = 0, 0
        bboxes = []

        if self.word_bbox:
            word_bboxes = []
            next_word_first_char = 0
            running_word_len = 0
            next_word_start_x = x_pos
        
        for i, c in enumerate(text):

            w, h = self.digital_font.getmask(c).size
            bottom_1 = self.digital_font.getsize(c)[1]
            bottom_2 = self.digital_font.getsize(text[:i+1])[1]
            bottom = bottom_1 if bottom_1 < bottom_2 else bottom_2
            x_jiggle = np.random.normal(self.char_dist, self.char_dist_std)
            offset = int(self.char_dist + x_jiggle)

            if c != "_":

                bbox = (x_pos, max(bottom - h, 0), w, h)
                bboxes.append(bbox)
                draw.text((x_pos, y_pos), c, font=self.digital_font, fill=1)
                x_pos += w + offset
                if safe_list_get(text, i+1, False) == "_" or i == len(text) - 1:
                    running_word_len += w
                else:
                    running_word_len += w + offset
                
            else:

                x_pos += w + offset
                if i == 0 and self.word_bbox:
                    next_word_start_x = x_pos
                if i != 0 and self.word_bbox:
                    _, wh = self.digital_font.getmask(text[next_word_first_char:i]).size
                    word_bottom = self.digital_font.getsize(text[next_word_first_char:i])[1]
                    next_word_first_char = i + 1
                    wbbx = (next_word_start_x, max(word_bottom - wh, 0), running_word_len, wh)
                    running_word_len = 0
                    next_word_start_x = x_pos
                    word_bboxes.append(wbbx)
                
        if self.word_bbox:
            if text[i] != "_":
                _, wh = self.digital_font.getmask(text[next_word_first_char:]).size
                word_bottom = self.digital_font.getsize(text[next_word_first_char:])[1]
                wbbx = (next_word_start_x, max(word_bottom - wh, 0), running_word_len, wh)
                word_bboxes.append(wbbx)
            return {"bboxes": bboxes, "word_bboxes": word_bboxes, "image": image}
        else:
            return {"bboxes": bboxes, "image": image}
    
    def generate_synthetic_textline_image_character_based(self, text):

        # create character renders
        char_renders = []
        for c in text:
            img = Image.new('RGB', (self.font_size*4, self.font_size*4), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((self.font_size, self.font_size), c, (255, 255, 255), 
                font=self.digital_font, anchor='mm')
            bbox = img.getbbox()
            if bbox is None:
                self.num_symbols -= 1
                continue
            x0, y0, x1, y1 = bbox
            pbbox = (x0, y0, x1, y1)
            char_render = ImageOps.invert(img.crop(pbbox))
            char_renders.append(char_render)

        # create canvas
        if not self.vertical:
            total_width = sum(cr.width for cr in char_renders)
            canvas_w = int((self.char_dist * (self.num_symbols + 1)) + total_width)
            canvas_h = int(self.font_size) 
            canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        else:
            total_height = sum(cr.height for cr in char_renders)
            canvas_h = int((self.char_dist * (self.num_symbols + 1)) + total_height)
            canvas_w = int(self.font_size) 
            canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        
        # pasting
        bboxes = []
        if self.vertical:
            y = self.char_dist
        else:
            x = self.char_dist
        
        for i in range(self.num_symbols):

            # get render
            curr_text = text[i]
            curr_render = char_renders[i]
            w, h = curr_render.size
            
            # account for spaces
            if curr_text == "_":
                if self.vertical:
                    y += h
                else:
                    x += w
                continue

            # create y offset
            if not self.vertical:
                height_diff = canvas_h - h
                if height_diff < 0: height_diff = 0
                y = height_diff // 2
                if curr_text in self.low_chars:
                    y = canvas_h - h - self.char_dist
            else:
                width_diff = canvas_w - w
                if width_diff < 0: width_diff = 0
                x = width_diff // 2

            # pasting!
            canvas.paste(curr_render, (x, y))
            bboxes.append((x, y, w, h))

            # move x position along
            jiggle = min(self.char_dist, abs(np.random.normal(0, self.char_dist_std)))
            if not self.vertical:                
                x += w + int(self.char_dist - jiggle)
            else:
                y += h + int(self.char_dist - jiggle)
        
        return {"bboxes": bboxes, "image": canvas}

    def generate_synthetic_textline(self, image_id):

        self.select_font()
        if self.single_words:
            textline_text = self.generate_synthetic_word_text()
        elif self.wiki_text:
            textline_text = self.generate_synthetic_wiki_text()
        else:
            textline_text = self.generate_synthetic_textline_text()

        if self.language == "jp" or self.language == "ja":
            out_dict = self.generate_synthetic_textline_image_character_based(textline_text)
        elif self.language == "en":
            out_dict = self.generate_synthetic_textline_image_latin_based(textline_text)
            
        out_image = self.synth_transform(out_dict["image"])
        image_name = f"{self.setname}_{image_id}.png"
        out_dict["trans_image"] = out_image
        out_dict["image_name"] = image_name
        out_dict["text"] = textline_text
        
        out_image.save(os.path.join(self.save_path, image_name))

        return out_dict

    def clean_wiki_text(self, x):
        clean_text = x.replace("\n", "").replace("=", "")
        clean_text = ''.join([i if (i in self.all_chars or i==" ") else '' for i in clean_text])
        return clean_text

    @staticmethod
    def wiki_check(random_page_name, min_size=50):
        try:
            random_page = wikipedia.page(random_page_name)
            if len(random_page.content) < min_size:
                return None
            else:
                return random_page
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            print("Retrying...")
            return None