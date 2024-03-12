import os
from glob import glob
import numpy as np
from PIL import Image


def wolf_threshold(input_path: str, output_path: str, window_size, k = 0.5, r = 128) -> None:
    if window_size % 2 == 0:
        window_size += 1

    input_img = Image.open(input_path)
    input_array = np.array(input_img)
    height, width = input_array.shape[:2]

    min_bright = np.amin(input_array)
    pad_img = np.pad(input_array, window_size, mode="reflect")

    output_array = np.zeros((height, width, input_array.shape[2]), dtype=input_array.dtype)

    for x in range (width):
        for y in range(height):
            local_window = pad_img[y:y + window_size, x:x + window_size]
            local_std = np.std(local_window)
            local_mean = np.mean(local_window)

            threshold = (1 - k) * local_mean + k * min_bright + k * (local_std / r) * (local_mean - min_bright)
            if input_array[y, x][0] > threshold:
                output_array[y, x] = 255
            else:
                output_array[y, x] = 0  
    
    new_img = Image.fromarray(output_array)
    new_img.save(output_path)
    print(f"Wrote threshold in {output_path}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    relative_path = "../1/output/*"
    for input_path in glob(relative_path):
        curr_opath = os.path.join(output_path, os.path.basename(input_path))
        wolf_threshold(input_path, curr_opath, 15)

if __name__ == "__main__":
    main()