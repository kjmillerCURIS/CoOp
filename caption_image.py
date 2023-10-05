
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap

#chatGPT wrote most of this function
def caption_image(numI, caption_lines, font_size=16, width_ratio=0.9, height_buffer=0.25):

    image = Image.fromarray(numI)

    # Define font and font size
    font = ImageFont.truetype('/usr/share/fonts/liberation/LiberationSans-Regular.ttf', font_size)
    
    # Calculate text height (per line) and set max num chars for wrapping
    probe_caption = ''.join(caption_lines)
    left, top, right, bottom = font.getbbox(probe_caption)
    text_height = bottom - top
    assert(text_height > 0)
    char_width = (right - left) / len(probe_caption)
    max_width = int(width_ratio * image.width / char_width)
    
    # Wrap caption text
    wrapper = textwrap.TextWrapper(width=max_width, break_long_words=False, drop_whitespace=False)
    wrapped_lines = []
    for caption in caption_lines:
        my_caption = caption
        if caption == '':
            my_caption = ' '

        wrapped_lines.extend(wrapper.wrap(my_caption))
    
    # Create new image with white background
    new_image = Image.new('RGB', (image.width, int(image.height + (1 + height_buffer) * (text_height * len(wrapped_lines)))), color=(255, 255, 255))
    
    # Paste original image on top
    new_image.paste(image, (0, 0))
    
    # Create a drawing object
    draw = ImageDraw.Draw(new_image)
    
    # Calculate text position
    text_x = int((1 - width_ratio) * new_image.width / 2)
    text_y = int(image.height + height_buffer * text_height * len(wrapped_lines) / 2)
    
    # Add wrapped text to image
    for line in wrapped_lines:
        draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))
        text_y += text_height
    
    numInew = np.array(new_image)
    return numInew

if __name__ == '__main__':
    caption_lines = ['Example caption that is too long to fit on one line, so it will be wrapped to multiple lines', '', ' '.join(['meow']*420)]
    numI = cv2.imread('meow.png')
    numInew = caption_image(numI, caption_lines)
    cv2.imwrite('meow_with_caption.png', numInew)
