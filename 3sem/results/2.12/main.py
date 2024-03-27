import os
from glob import glob
import numpy as np
from PIL import Image, ImageChops


def erosion(image_arr, kernel):
    height, width = image_arr.shape[:2]
    ker_height, ker_width = kernel.shape

    new_height, new_width = ker_height // 2, ker_width // 2
    pad_img = np.pad(image_arr, ((new_height, new_height), (new_width, new_width)), mode='constant', constant_values=0)

    output_array = np.zeros((height, width), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            output_array[y, x] = np.min(pad_img[y:y + new_height, x: x + new_width] * kernel)

    return output_array

def dilation(image_arr, kernel):
    height, width = image_arr.shape[:2]
    ker_height, ker_width = kernel.shape

    new_height, new_width = ker_height // 2, ker_width // 2
    pad_img = np.pad(image_arr, (new_height, new_height), mode='constant', constant_values=0)

    output_array = np.zeros((height, width), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            output_array[y, x] = np.max(image_arr[y:y + new_height, x: x + new_width] * kernel)

    return output_array

def diffmap(old_img, new_img):
    diff_array = ImageChops.difference(old_img, new_img)
    res = ImageChops.invert(diff_array)
    return res

def closing(input_path: str, output_path: str, diff_path: str) -> None:
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    input_img = Image.open(input_path).convert('RGB')
    input_array = np.array(input_img).astype(np.uint8)

    tmp_array = dilation(input_array, kernel)
    output_array = erosion(tmp_array, kernel)

    new_img = Image.fromarray(output_array.astype(np.uint8), 'L').convert('RGB')
    new_img.save(output_path + ".png")
    print(f"Wrote closing in {output_path}.png")

    diff_img = diffmap(input_img, new_img)
    diff_img.save(diff_path + ".png")
    print(f"Wrote diffmap in {diff_path}.png")
    

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    relative_path = "../1/output/*"
    for input_path in glob(relative_path):
        curr_opath = os.path.join(output_path, os.path.splitext("closing_" + os.path.basename(input_path))[0])
        diff_opath = os.path.join(output_path, os.path.splitext("diff_" + os.path.basename(input_path))[0])
        closing(input_path, curr_opath, diff_opath)

if __name__ == "__main__":
    main()