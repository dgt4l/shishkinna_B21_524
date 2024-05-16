from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt

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

def write_data():
    with open('letters.csv', 'w', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['letter', 'weight', 'rel_weight', 'center', 
                                                  'rel_center', 'inertia', 'rel_inertia'])
        writer.writeheader()
        letters = "𐎀𐎁𐎂𐎃𐎄𐎅𐎆𐎇𐎈𐎉𐎊𐎋𐎌𐎍𐎎𐎏𐎐𐎑𐎒𐎓𐎔𐎕𐎖𐎗𐎘𐎙𐎚𐎛𐎜𐎝"

        for letter in letters:
            img = Image.open(f"fonts/letters/{letter}.png")
            
            data = get_data(img)
            data['letter'] = letter

            writer.writerow(data)

def main():
    save_profiles()
    write_data()


if __name__ == '__main__':
    main()

