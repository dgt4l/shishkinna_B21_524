from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from math import ceil

ugaritic_unicode = ["10380", "10381", "10382", "10383", "10384", "10385", "10386", "10387", "10388",
                    "10389", "1038A", "1038B", "1038C", "1038D", "1038E", "1038F", "10390", "10391",
                    "10392", "10393", "10394", "10395", "10396", "10397", "10398", "10399", "1039A",
                    "1039B", "1039C", "1039D"]

ugaritic_letters = [chr(int(letter, 16)) for letter in ugaritic_unicode]

def find_first_nonzero(ar):
    return np.min(np.nonzero(ar))

def find_last_nonzero(ar):
    return np.max(np.nonzero(ar)) + 1

def find_first_last_non_zero(ar):
    return find_first_nonzero(ar), find_last_nonzero(ar)

def convert_to_bd_image(image):
    image = np.array(image)
    return 1 - image / 255

def calc_horizontal_profile(image):
    return convert_to_bd_image(image).sum(axis=1)

def calc_vertical_profile(image):
    return convert_to_bd_image(image).sum(axis=0)

def calc_profiles(image):
    return calc_horizontal_profile(image), calc_vertical_profile(image)

def generate_font_images(font_path, font_size):
    letters = ugaritic_letters
    font = ImageFont.truetype(font_path, font_size)

    for letter in letters:
        _, _, width, height = font.getbbox(letter)

        image = Image.new("L", (width, height), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), letter, font=font, color="black")

        horizontal_profile, vertical_profile = calc_profiles(image)

        upper, lower = find_first_last_non_zero(horizontal_profile)
        left, right = find_first_last_non_zero(vertical_profile)

        cropped_letter = image.crop((left, upper, right, lower))

        cropped_letter.save(f"letters/{letter}.bmp")

def main():
    generate_font_images("NotoSansUgaritic-Regular.ttf", 100)

if __name__ == '__main__':
    main()