import os
from glob import glob
import numpy as np
from PIL import Image

def resampling(input_path: str, output_path: str, m, n) -> None:
    input_img = Image.open(input_path)
    tmp_img = upsampling(input_img, n)
    output_img = downsampling(tmp_img, m)

    output_img.save(output_path)
    print(f"Wrote resampling in {output_path}")

def upsampling(input_image: Image, m):
    input_array = np.array(input_image)
    height, width = input_array.shape[:2]
    new_w = width * m 
    new_h = height * m 
    output_array = np.zeros((new_h, new_w, input_array.shape[2]), dtype=input_array.dtype)

    for x in range (new_w):
        for y in range(new_h):
            orig_x = x // m
            orig_y = y // m
            output_array[y, x] = input_array[orig_y, orig_x]

    new_img = Image.fromarray(output_array)
    
    return new_img

def downsampling(tmp_image: Image, n):
    tmp_array = np.array(tmp_image)
    height, width = tmp_array.shape[:2]
    new_w = width // n
    new_h = height // n
    output_array = np.zeros((new_h, new_w, tmp_array.shape[2]), dtype=tmp_array.dtype)

    for x in range (new_w):
        for y in range(new_h):
            orig_x = x * n
            orig_y = y * n
            output_array[y, x] = tmp_array[orig_y, orig_x]

    new_img = Image.fromarray(output_array)
    
    return new_img


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    relative_path = "../input/*"
    m, n = list(map(int, input("Enter the degrees of resampling\n").split()))
    for input_path in glob(relative_path):
        curr_opath = os.path.join(output_path, os.path.basename(input_path))
        resampling(input_path, curr_opath, n, m)

if __name__ == "__main__":
    main()