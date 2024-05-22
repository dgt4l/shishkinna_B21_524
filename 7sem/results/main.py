from PIL import Image, ImageFont, ImageDraw
import numpy as np
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce


def blackened(image):
    image = np.array(image)
    return 1 - image / 255

def first_nonzero(ar):
    return np.min(np.nonzero(ar))

def last_nonzero(ar):
    return np.max(np.nonzero(ar)) + 1

def first_last_non_zero(ar):
    return first_nonzero(ar), last_nonzero(ar)

def horizontal_profile(image):
    return blackened(image).sum(axis=1)

def vertical_profile(image):
    return blackened(image).sum(axis=0)

def profiles(image):
    return horizontal_profile(image), vertical_profile(image)

def generate_sentence(sentence, input_path, font_size, number):
    font = ImageFont.truetype(input_path, font_size)

    width = reduce(
        lambda acc, curr: acc + curr,
        map(lambda symbol: font.getbbox(symbol)[2], sentence), 0)

    height = max(map(lambda symbol: font.getbbox(symbol)[3], sentence))

    image = Image.new("L", (width + 20, height + 20), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), sentence, font=font, color="black")
    image.save(f"output/sentence_{number}.bmp")

def find_letter_ranges(y_profile, horizontal_border):
    ranges = []

    left = 0

    while left < y_profile.size:
        while left < y_profile.size and y_profile[left] == 0:
            left  += 1

        right = left 

        while right < y_profile.size and y_profile[right] != 0:
            right += 1
        if (left != right):
            ranges.append((left, right))
        left = right

    whitespace_ranges = []
    for symbol_range in ranges:
        if len(whitespace_ranges) == 0:
            whitespace_ranges.append(symbol_range)
            continue
        if (symbol_range[0] - whitespace_ranges[-1][1] < horizontal_border):
            whitespace_ranges[-1] = (whitespace_ranges[-1][0], symbol_range[1])
        else:
            whitespace_ranges.append(symbol_range)
    
    return whitespace_ranges

def borders_from_ranges(image, ranges):
    borders = []
    _, height = image.size

    for i, letter_range in enumerate(ranges):
        left, right = letter_range
        symbol = image.crop((left, 0, right, height))

        x_profile = horizontal_profile(symbol)

        width, _ = symbol.size
        lower, upper = first_last_non_zero(x_profile)
        symbol = symbol.crop((0, lower, width, upper))

        borders.append((left, lower, right, upper))

    return borders

def get_segments(image, x_border):
    _, y_profile = profiles(image)
    
    ranges = find_letter_ranges(y_profile, x_border)
    segments = borders_from_ranges(image, ranges)

    return segments

def calc_black_weight(img_arr):
    return np.sum(img_arr)

def calc_rel_black_weight(img_arr):
    return calc_black_weight(img_arr) / img_arr.size

def calc_center(img_arr):
    height, width = img_arr.shape

    black_weight = calc_black_weight(img_arr)

    center_x = (np.sum(img_arr, axis=1) @ np.array(range(height))) / black_weight
    center_y = (np.sum(img_arr, axis=0) @ np.array(range(width))) / black_weight

    return center_x, center_y

def calc_rel_center(img_arr):
    height, width = img_arr.shape

    center_x, center_y = calc_center(img_arr)

    return (center_x - 1) / (height - 1), (center_y - 1) / (width - 1)

def calc_x_inertia(img_arr):
    _, width = img_arr.shape
    _, y_center = calc_center(img_arr)

    return np.sum((np.array(range(width)) - y_center)**2 @ np.transpose(img_arr))

def calc_y_inertia(img_arr):
    height, _ = img_arr.shape
    x_center, _ = calc_center(img_arr)
    return np.sum((np.array(range(height)) - x_center)**2 @ img_arr)

def calc_rel_x_inertia(img_arr):
    height, width = img_arr.shape

    return calc_x_inertia(img_arr) / (height**2 * width**2)

def calc_rel_y_inertia(img_arr):
    height, width = img_arr.shape

    return calc_y_inertia(img_arr) / (height**2 * width**2)

def create_features(letters):
    report = {
        "Symbol": [],
        "Black Mass": [],
        "Center of Gravity, X": [],
        "Center of Gravity, Y": [],
        "Horizontal Moment of Inertia": [],
        "Vertical Moment of Inertia": [],

    }

    for letter in letters:
        img = Image.open(f"../../5sem/results/fonts/letters/{letter}.bmp")
        b_img = blackened(img)

        x_center, y_center = calc_rel_center(b_img)

        report["Symbol"].append(letter)
        report["Black Mass"].append(calc_rel_black_weight(b_img))
        report["Center of Gravity, X"].append(x_center)
        report["Center of Gravity, Y"].append(y_center)
        report["Horizontal Moment of Inertia"].append(calc_rel_x_inertia(b_img))
        report["Vertical Moment of Inertia"].append(calc_rel_y_inertia(b_img))
    
    df = pd.DataFrame(report)

    return df

def normalize(value, params):
    col_min, col_max = params
    return (value - col_min) / (col_max - col_min)

def normalize_features(features):
    normalized_features = features.copy()

    feature_to_norm_params = dict()

    for column in normalized_features:
        if column == "Symbol":
            continue

        col_min = normalized_features[column].min()
        col_max = normalized_features[column].max()

        feature_to_norm_params[column] = col_min, col_max

        normalized_features[column] = (normalized_features[column] - col_min) / (col_max - col_min)
    
    return normalized_features, feature_to_norm_params

def recognite_letter(b_letter, feature_matrix, norm_params, letters):
    x_center, y_center = calc_rel_center(b_letter)

    letter_features = np.array([
        normalize(calc_rel_black_weight(b_letter), norm_params["Black Mass"]),
        normalize(x_center, norm_params["Center of Gravity, X"]),
        normalize(y_center, norm_params["Center of Gravity, Y"]),
        normalize(calc_rel_x_inertia(b_letter), norm_params["Horizontal Moment of Inertia"]),
        normalize(calc_rel_y_inertia(b_letter), norm_params["Vertical Moment of Inertia"]),
    ])

    distances = np.sqrt(np.sum((feature_matrix - letter_features)**2, axis=1))
    probs = np.exp(-distances)

    result = []

    for i, letter in enumerate(letters):
        result.append((letter, probs[i]))

    return sorted(result, key=lambda x: -x[1])

def to_matrix():
    ugaritic_letters = "𐎀𐎁𐎂𐎃𐎄𐎅𐎆𐎇𐎈𐎉𐎊𐎋𐎌𐎍𐎎𐎏𐎐𐎑𐎒𐎓𐎔𐎕𐎖𐎗𐎘𐎙𐎚𐎛𐎜𐎝"
    features = create_features(ugaritic_letters)
    norm_features, norm_params = normalize_features(features)
    numeric_features = [
        "Black Mass",
        "Center of Gravity, X",
        "Center of Gravity, Y",
        "Horizontal Moment of Inertia",
        "Vertical Moment of Inertia"]

    feature_matrix = norm_features[numeric_features].to_numpy()
    return feature_matrix, norm_params

def sentence_recognition(sentence, flag, size):

    generate_sentence(sentence, "NotoSansUgaritic-Regular.ttf", size, flag)
    
    ugaritic_letters = "𐎀𐎁𐎂𐎃𐎄𐎅𐎆𐎇𐎈𐎉𐎊𐎋𐎌𐎍𐎎𐎏𐎐𐎑𐎒𐎓𐎔𐎕𐎖𐎗𐎘𐎙𐎚𐎛𐎜𐎝"
    image = Image.open(f"output/sentence_{flag}.bmp").convert("L")
    new_image = blackened(image)
    borders = get_segments(image, 4)
    recognited = []
    feature_data, norm_data = to_matrix()
    for border in borders:
        left, lower, right, upper = border
        black_letter = new_image[lower:upper, left:right]
        recognited.append(
            recognite_letter(black_letter, feature_data, norm_data, ugaritic_letters))

    sentence.replace(" ", "")
    recognited_sentence = ''.join(letter[0][0] + " " for letter in recognited)
    with open(f"output/result_{flag}.txt", "w") as f:
        for i, recognitions in enumerate(recognited):
            f.write(f"{i+1}: {recognitions}\n")

    print(f"{recognited_sentence}, {len(sentence)} : {len(recognited_sentence) // 2}")

def main():
    sentences = ["𐎀𐎁𐎂𐎃𐎄𐎅𐎆𐎇𐎈𐎊", "𐎖𐎗𐎘𐎙𐎚𐎛𐎜𐎝", "𐎀𐎁𐎂𐎃𐎄𐎅𐎆𐎇𐎈𐎉𐎊𐎋𐎌𐎍𐎎𐎏𐎐𐎑𐎒𐎓𐎔𐎕𐎖𐎗mm𐎘𐎙𐎚𐎛𐎜𐎝"]
    for i, sentence in enumerate(sentences):
        size = 50
        sentence_recognition(sentence, i+1, size)

if __name__ == "__main__":
    main()
    
