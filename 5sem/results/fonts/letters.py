from PIL import Image, ImageDraw, ImageFont
import numpy as np


def binarization(img, threshold = 100):
    img_arr = np.array(img)
    bin_img = np.zeros(shape=img_arr.shape)
    bin_img[img_arr > threshold] = 255
    res = Image.fromarray(bin_img.astype(np.uint8), 'L')
    return res


def main():
    letters = "𐎀𐎁𐎂𐎃𐎄𐎅𐎆𐎇𐎈𐎉𐎊𐎋𐎌𐎍𐎎𐎏𐎐𐎑𐎒𐎓𐎔𐎕𐎖𐎗𐎘𐎙𐎚𐎛𐎜𐎝"
    font = ImageFont.truetype("NotoSansUgaritic-Regular.ttf", 100)

    for letter in letters:
        _, _, width, height = font.getbbox(letter) 

        img = Image.new("L", (width, height), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), letter, font=font, color="black")
        res = binarization(img)
        res.save(f"letters/{letter}.png")

if __name__ == '__main__':
    main()