import os
from glob import glob
import numpy as np
from PIL import Image


def semitone(img):
    height = img.shape[0]
    width = img.shape[1]

    new_img = np.zeros(shape=(height, width))

    for x in range(width):
        for y in range(height):
            pixel = img[y, x]
            new_img[y, x] = 0.3 * pixel[0] + 0.59 * pixel[1] + 0.11 * pixel[2]

    return new_img.astype(np.uint8)

def dilation(image, kernel, output_path: str):
    height = kernel.shape[0]
    width = kernel.shape[1]
    pad_height = height // 2
    pad_width = width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
                          constant_values=255)

    res_image = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            min_value = np.min(padded_image[y:y + height, x:x + width] * kernel)

            res_image[y, x] = 0 if min_value == 0 else 255

    Image.fromarray(res_image).convert('RGB').save(output_path + '.png')

    return res_image

def diffmap(old_arr, new_arr, output_path:str):
    res_arr = old_arr - new_arr
    res = Image.fromarray(np.uint8(res_arr))
    res.save(output_path + '.png')
    print(f"Wrote contours in {output_path}.png")

def binarization(input_path: str, output_path: str, threshold = 100):
    img = Image.open(input_path).convert('RGB')
    img_array = semitone(np.array(img))
    new_img = np.zeros(shape=img_array.shape)
    new_img[img_array > threshold] = 255
    output = Image.fromarray(np.uint8(new_img))
    output.save(output_path + '.png')
    print(f"Wrote binarization in {output_path}.png")
    return new_img


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    relative_path = "input/*"
    for input_path in glob(relative_path):
        kernel = np.ones((3, 3), dtype=np.uint8)

        bin_opath = os.path.join(output_path, os.path.splitext("bin_" + os.path.basename(input_path))[0])
        dilation_opath = os.path.join(output_path, os.path.splitext("dilation_" + os.path.basename(input_path))[0])
        contour_opath = os.path.join(output_path, os.path.splitext("contour_" + os.path.basename(input_path))[0])

        bin_img = binarization(input_path, bin_opath)
        dilate_img = dilation(bin_img, kernel, dilation_opath)
        diffmap(bin_img, dilate_img, contour_opath)

    
if __name__ == "__main__":
    main()

