import os
from glob import glob
import numpy as np
from PIL import Image

def upsampling(input_path: str, output_path: str, degree: int) -> None:
    input_img = Image.open(input_path)
    input_array = np.array(input_img)
    height, width = input_array.shape[:2]
    new_w = width * degree 
    new_h = height * degree 
    output_array = np.zeros((new_h, new_w, input_array.shape[2]), dtype=input_array.dtype)

    for x in range (new_w):
        for y in range(new_h):
            orig_x = x // degree
            orig_y = y // degree
            output_array[y, x] = input_array[orig_y, orig_x]

    new_img = Image.fromarray(output_array)
    new_img.save(output_path)
    print(f"Wrote upsampling in {output_path}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    relative_path = "../input/*"
    degree = int(input("Enter the degree of upsampling\n"))
    for input_path in glob(relative_path):
        curr_opath = os.path.join(output_path, os.path.basename(input_path))
        upsampling(input_path, curr_opath, degree)

if __name__ == "__main__":
    main()