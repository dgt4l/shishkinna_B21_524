import os
from glob import glob
import numpy as np
from PIL import Image

def resampling(input_path: str, output_path: str, n, m) -> None:
    img = Image.open(input_path)
    img_array = np.array(img)

    height, width = img_array.shape[:2]
    
    new_w = width * m // n
    new_h = height * m // n

    output_array = np.zeros((new_h, new_w, img_array.shape[2]), dtype=img_array.dtype)

    for y in range (new_h):
        for x in range(new_w):
            orig_y = n * y // m
            orig_x = n * x // m
            output_array[y, x] = img_array[orig_y, orig_x]

    new_img = Image.fromarray(output_array)
    new_img.save(output_path)
    print(f"Wrote resampling in {output_path}")


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