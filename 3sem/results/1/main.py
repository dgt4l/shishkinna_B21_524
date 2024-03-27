import os
from glob import glob
import numpy as np
from PIL import Image

def semitone(input_path: str, output_path: str) -> None:
    input_img = Image.open(input_path)
    input_array = np.array(input_img)
    height, width = input_array.shape[:2]
    output_array = np.zeros((height, width, input_array.shape[2]), dtype=input_array.dtype)

    for x in range (width):
        for y in range(height):
            pixel = input_array[y, x]
            output_array[y, x] = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]

    new_img = Image.fromarray(output_array)
    new_img.save(output_path + ".png")
    print(f"Wrote semitone in {output_path}.png")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    relative_path = "input/*"
    for input_path in glob(relative_path):
        curr_opath = os.path.join(output_path, os.path.splitext("semitone_" + os.path.basename(input_path))[0])
        semitone(input_path, curr_opath)

if __name__ == "__main__":
    main()