import numpy as np
import csv
from PIL import Image, ImageFont, ImageDraw
from functools import reduce

ugaritic_unicode = ["10380", "10381", "10382", "10383", "10384", "10385", "10386", "10387", "10388",
                    "10389", "1038A", "1038B", "1038C", "1038D", "1038E", "1038F", "10390", "10391",
                    "10392", "10393", "10394", "10395", "10396", "10397", "10398", "10399", "1039A",
                    "1039B", "1039C", "1039D"]

ugaritic_letters = [chr(int(letter, 16)) for letter in ugaritic_unicode]

def convert_to_bd_image(image):
    image = np.array(image)
    return 1 - image / 255

def calc_black_weight(bd_image):
    return np.sum(bd_image)

def calc_rel_black_weight(bd_image):
    return calc_black_weight(bd_image) / bd_image.size

def calc_center_of_gravity(bd_image):
    height, width = bd_image.shape

    black_weight = calc_black_weight(bd_image)

    center_x = (np.sum(bd_image, axis=1) @ np.array(range(height))) / black_weight
    center_y = (np.sum(bd_image, axis=0) @ np.array(range(width))) / black_weight

    return center_x, center_y

def calc_rel_center_of_gravity(bd_image):
    height, width = bd_image.shape

    center_x, center_y = calc_center_of_gravity(bd_image)

    return (center_x - 1) / (height - 1), (center_y - 1) / (width - 1)

def calc_horizontal_inertia_moment(bd_image):
    _, width = bd_image.shape
    _, y_center = calc_center_of_gravity(bd_image)

    return np.sum((np.array(range(width)) - y_center)**2 @ np.transpose(bd_image))

def calc_vertical_inertia_moment(bd_image):
    height, _ = bd_image.shape
    x_center, _ = calc_center_of_gravity(bd_image)
    return np.sum((np.array(range(height)) - x_center)**2 @ bd_image)

def calc_rel_horizontal_inertia_moment(bd_image):
    height, width = bd_image.shape

    return calc_horizontal_inertia_moment(bd_image) / (height**2 * width**2)

def calc_rel_vertical_inertia_moment(bd_image):
    height, width = bd_image.shape

    return calc_vertical_inertia_moment(bd_image) / (height**2 * width**2)

def create_features_report(symbols):
    with open('letters.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Symbol', 'Black Mass', 'Center of Gravity, X', 'Center of Gravity, Y',
                    'Horizontal Moment of Inertia','Vertical Moment of Inertia']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for symbol in symbols:
            report = {
            "Symbol": [],
            "Black Mass": [],
            "Center of Gravity, X": [],
            "Center of Gravity, Y": [],
            "Horizontal Moment of Inertia": [],
            "Vertical Moment of Inertia": []
            }
            image = Image.open(f"fonts/letters/{symbol}.bmp")
            bd_image = convert_to_bd_image(image)

            x_center, y_center = calc_rel_center_of_gravity(bd_image)

            report["Symbol"].append(symbol)
            report["Black Mass"].append(calc_rel_black_weight(bd_image))
            report["Center of Gravity, X"].append(x_center)
            report["Center of Gravity, Y"].append(y_center)
            report["Horizontal Moment of Inertia"].append(calc_rel_horizontal_inertia_moment(bd_image))
            report["Vertical Moment of Inertia"].append(calc_rel_vertical_inertia_moment(bd_image))

            writer.writerow(report)




def main():
    create_features_report(ugaritic_letters)

if __name__ == "__main__":
    main()
