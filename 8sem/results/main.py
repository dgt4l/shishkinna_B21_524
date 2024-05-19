import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from glob import glob

def semitone(img):
    height = img.shape[0]
    width = img.shape[1]

    new_img = np.zeros(shape=(height, width))

    for x in range(width):
        for y in range(height):
            pixel = img[y, x]
            new_img[y, x] = 0.3 * pixel[0] + 0.59 * pixel[1] + 0.11 * pixel[2]

    return new_img.astype(np.uint8)

def haralic(img, d = 1):
    width, height = img.shape
    size = 256
    haralic = np.zeros((size, size))
    
    for x in range(d, width - d):
        for y in range(d, height - d):
            haralic[img[x + d, y], img[x, y]] += 1
            haralic[img[x - d, y], img[x, y]] += 1
            haralic[img[x, y + d], img[x, y]] += 1
            haralic[img[x, y - d], img[x, y]] += 1
    return np.uint8(haralic)

def asm(matrix):
    return np.sum(np.square(matrix))

def mpr(matrix):
    return np.max(matrix)

def ent(matrix):
    ent = 0
    width, height = matrix.shape

    for x in range(width):
        for y in range(height):
            if matrix[x, y] > 0:
                ent -= matrix[x, y] * np.log2(matrix[x, y])
    return ent

def tr(matrix):
    return np.trace(matrix)

def contrast(img):
    flat_img = img.flatten()
    mean = round(np.mean(flat_img))
    l = 5

    positive_div = max(2, max(flat_img) - mean)
    negative_div = max(2, mean - min(flat_img))
    
    positive_alpha = 2 ** (l - 1) / np.log(positive_div)
    negative_alpha = 2 ** (l - 1) / np.log(negative_div)

    res = np.zeros_like(img)
    width, height = img.shape

    for x in range(width):
        for y in range(height):
            f = img[x, y] - mean
            if f >= 1:
                res[x, y] = mean + positive_alpha * np.log(f)
            elif f <= -1:
                res[x, y] = mean - negative_alpha * np.log(np.abs(f))
            else:
                res[x, y] = mean
    return res

def hist(matrix, output_path):
    shape = np.reshape(matrix, (1, -1))
    plt.figure()
    plt.hist(shape[0], bins=256)
    plt.savefig(output_path)

def process_images():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    relative_path = "input/*"
    for input_path in glob(relative_path):

        img = Image.open(input_path)
        img_arr = np.array(img).astype(np.uint8)
        semi_arr = semitone(img_arr)
        contrast_arr = contrast(semi_arr)
        haralic_arr = haralic(semi_arr)
        haralic_contrast_arr = haralic(contrast_arr)
        
        semi_opath = os.path.join(output_path, os.path.splitext("semi_" + os.path.basename(input_path))[0])
        contrast_opath = os.path.join(output_path, os.path.splitext("contrast_" + os.path.basename(input_path))[0])
        haralic_opath = os.path.join(output_path, os.path.splitext("haralic_" + os.path.basename(input_path))[0])
        haralic_contrast_opath = os.path.join(output_path, os.path.splitext("haralic_contrast" + os.path.basename(input_path))[0])
        semi_hist = os.path.join(output_path, os.path.splitext("semi_hist" + os.path.basename(input_path))[0])
        contrast_hist = os.path.join(output_path, os.path.splitext("contrast_hist" + os.path.basename(input_path))[0])

        semi_img = Image.fromarray(semi_arr).save(semi_opath + ".png")
        contrast_img = Image.fromarray(contrast_arr).save(contrast_opath + ".png")
        haralic_img = Image.fromarray(haralic_arr).save(haralic_opath + ".png")
        haralic_contrast_img = Image.fromarray(haralic_contrast_arr).save(haralic_contrast_opath + ".png")

        hist(semi_arr, semi_hist)
        hist(contrast_arr, contrast_hist)

def main():
    process_images()


if __name__ == '__main__':
    main()

        

