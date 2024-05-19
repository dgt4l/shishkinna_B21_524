from PIL import Image, ImageFont, ImageDraw
import numpy as np
import csv
from math import sqrt
import matplotlib.pyplot as plt


def binarization(img, threshold = 100):
    img_arr = np.array(img)
    bin_img = np.zeros(shape=img_arr.shape)
    bin_img[img_arr > threshold] = 255
    res = Image.fromarray(bin_img.astype(np.uint8), 'L')
    return res

def generate_phrase():
    letters = "𐎀𐎁𐎂𐎃𐎄𐎅𐎆𐎇𐎈"
    font = ImageFont.truetype("NotoSansUgaritic-Regular.ttf", 100)
    _, _, width, height = font.getbbox(letters)
    img = Image.new("L", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), letters, font=font, color="black")
    res = binarization(img)
    res.save(f"output/sentence.png")

def black_white(img):
    return np.asarray(np.asarray(img) < 1, dtype=np.uint8)

def calculate_segments(img):
    img_arr = black_white(img)
    x_profiles = np.sum(img_arr, axis=0)
    lst = [] 
    new_lst = []  
    for i in range(len(x_profiles)):   
        if x_profiles[i] == 0:
            lst.append(i)
    lst.append(img.width)  

    for i in range(len(lst)-1):
        if lst[i] + 1 != lst[i+1]:
            new_lst.append(lst[i])
            new_lst.append(lst[i+1])
    new_lst.append(img.width-1)
    new_lst = sorted(list(set(new_lst))) 
    
    segments = []
    for i in range(0, len(new_lst)-1, 2):
        segments.append((new_lst[i], new_lst[i+1]))
    return segments

def draw_segments():
    img = Image.open("output/sentence.png").convert('L')
    segments = calculate_segments(img)
    for i, segment in enumerate(segments):
        box = (segment[0], 0, segment[1] + 1, img.height)
        res = img.crop(box)
        res.save(f"output/letters/{i + 1}.png")

def black_white(img):
    return np.asarray(np.asarray(img) < 1, dtype=np.uint8)

def calculate_black_weight(img_arr):
    height, width = img_arr.shape
    weight = 0

    for x in range(width):
        for y in range(height):
            if img_arr[y, x] == 0:
                weight += 1

    return weight, weight / img_arr.size

def calculate_center(img_arr):
    height, width = img_arr.shape
    x_coord, y_coord = 0, 0

    for x in range(width):
        for y in range(height):
            if img_arr[y, x] == 0:
                x_coord += x
                y_coord += y

    x_center = x_coord / img_arr.size
    y_center = y_coord / img_arr.size

    return (x_center, y_center), (x_center / width, y_center / height)

def calculate_moments_of_inertia(img_arr, x_center, y_center):
    height, width = img_arr.shape
    x_inertia, y_inertia = 0, 0

    for x in range(width):
        for y in range(height):
            x_inertia = (y - y_center) ** 2
            y_inertia = (x - x_center) ** 2

    x_inertia_norm = x_inertia / (width ** 2 * height ** 2)
    y_inertia_norm = y_inertia / (width ** 2 * height ** 2)

    return (x_inertia, y_inertia), (x_inertia_norm, y_inertia_norm)

def get_profiles(img):
    img_arr = black_white(img)
    return {
            'x': np.sum(img_arr, axis=0),
            'x_r': np.arange(start=1, stop=img_arr.shape[1] + 1).astype(int),
            'y_r': np.arange(start=1, stop=img_arr.shape[0] + 1).astype(int),
            'y': np.sum(img_arr, axis=1)
        }

def write_profiles(img, output_path, type='x'):
    profiles = get_profiles(img)
    
    if type == 'x':
        plt.bar(x=profiles['x_r'], height=profiles['x'], width=0.9)
        plt.ylim(0, 60)
    
    else:
        plt.barh(y=profiles['y_r'], width=profiles['y'], height=0.9)
        plt.ylim(60, 0)

    plt.savefig(output_path)
    plt.clf()
    
def save_profiles():
    letters = "𐎀𐎁𐎂𐎃𐎄𐎅𐎆𐎇𐎈𐎉𐎊𐎋𐎌𐎍𐎎𐎏𐎐𐎑𐎒𐎓𐎔𐎕𐎖𐎗𐎘𐎙𐎚𐎛𐎜𐎝"
    for letter in letters:
        img = Image.open(f"fonts/letters/{letter}.png")
        x_path = f"profiles/{letter}_x.png"
        y_path = f"profiles/{letter}_y.png"
        write_profiles(img, x_path, 'x')
        write_profiles(img, y_path, 'y')

def get_data(img):
    img_arr = np.array(img).astype(np.uint8)

    weight, norm_weight = calculate_black_weight(img_arr)
    center, norm_center = calculate_center(img_arr)
    inertia, norm_inertia = calculate_moments_of_inertia(img_arr, center[0], center[1])

    return {
        'weight': weight,
        'rel_weight': norm_weight,
        'center': center,
        'rel_center': norm_center,
        'inertia': inertia,
        'rel_inertia': norm_inertia
    }

def load_data_from_csv(path):
    with open(path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        result = {}
        for row in reader:
            result[row['letter']] = {
                'rel_weight': float(row['rel_weight']), 
                'rel_center': tuple(map(float, row['rel_center'][1:len(row['rel_center'])-1].split(', '))),
                'rel_inertia': tuple(map(float, row['rel_inertia'][1:len(row['rel_inertia'])-1].split(', ')))
            }
        return result

def calculate_distance(data1, data2):
    distance = sqrt(data1['rel_weight'] - data2['rel_weight']**2 +
        (data1['rel_center'][0] - data2['rel_center'][0])**2 +
        (data1['rel_center'][1] - data2['rel_center'][1])**2 +
        (data1['rel_inertia'][0] - data2['rel_inertia'][0])**2 +
        (data1['rel_inertia'][1] - data2['rel_inertia'][1])**2
    )

    return distance

def calculate_all_distances(letters_data, sentence_data):
    result = {}
    for letter, data in letters_data.items():
        result[letter] = calculate_distance(sentence_data, data)

    _max = max(result.values())

    new_result = {}
    for letter, distance in result.items():
        new_result[letter] = (_max - distance) / _max

    return new_result


def main():
    # generate_phrase()
    # draw_segments()
    results = load_data_from_csv("letters.csv")
    img = Image.open("output/letters/1.png")
    data = get_data(img)
    # print(data)
    # print(results["𐎁"])
    result = calculate_all_distances(results, data)
    print(result)

if __name__ == '__main__':
    main()