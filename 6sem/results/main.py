from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


def binarization(img, threshold = 100):
    img_arr = np.array(img)
    bin_img = np.zeros(shape=img_arr.shape)
    bin_img[img_arr > threshold] = 255
    res = Image.fromarray(bin_img.astype(np.uint8), 'L')
    return res

def generate_phrase():
    letters = "𐎀𐎁𐎂𐎃𐎄𐎅𐎆𐎇𐎈"
    font = ImageFont.truetype("NotoSansUgaritic-Regular.ttf", 100)
    _, _, width, height = font.getbbox(letters)
    img = Image.new("L", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), letters, font=font, color="black")
    res_img = binarization(img)
    res_img.save(f"output/sentence.png")

def black_white(img):
    return np.asarray(np.asarray(img) < 1, dtype=np.uint8)

def calculate_segments(img):
    img_arr = black_white(img)
    x_profiles = np.sum(img_arr, axis=0)
    lst = [] 
    new_lst = []  
    for i in range(len(x_profiles)):   
        if x_profiles[i] == 0:
            lst.append(i)
    lst.append(img.width)  

    for i in range(len(lst)-1):
        if lst[i] + 1 != lst[i+1]:
            new_lst.append(lst[i])
            new_lst.append(lst[i+1])
    new_lst.append(img.width-1)
    new_lst = sorted(list(set(new_lst))) 
    
    segments = []
    for i in range(0, len(new_lst)-1, 2):
        segments.append((new_lst[i], new_lst[i+1]))
    return segments

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
    img = Image.open(f"output/sentence.png")
    x_path = f"output/profiles/sentence_x.png"
    y_path = f"output/profiles/sentence_y.png"
    write_profiles(img, x_path, 'x')
    write_profiles(img, y_path, 'y')


def draw_segments():
    img = Image.open("output/sentence.png").convert('L')
    segments = calculate_segments(img)
    for i, segment in enumerate(segments):
        box = (segment[0] + 1, 0, segment[1] - 1, img.height)
        res = img.crop(box)
        res.save(f"output/letters/{i + 1}.png")


def main():
    generate_phrase()
    draw_segments()
    save_profiles()


if __name__ == '__main__':
    main()