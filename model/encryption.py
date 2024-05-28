from PIL import Image, ImageOps
import base64
from io import BytesIO

def imageToBase64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode('utf-8')
    return img_str

def base64toImage(base64string):
    img_str = base64.b64decode(base64string)
    image = Image.open(BytesIO(img_str))
    return image

def hide_message(message, image):
    print('began')
    message += '\0'
    newImage = image[:]
    num_bits = 1
    while len(message) * 8 > (len(image) - 2) * num_bits:
        num_bits += 1
    if num_bits == 3:
        num_bits += 1
    if num_bits > 4:
        return "Error: Message is too long"
    newImage[0] = ((image[0] & 0b11111000) | num_bits)

    position = 1
    for i in range(len(message) * 8 // num_bits):
        newImage[position] = (image[position] & (~((1 << num_bits) - 1))) | (ord(message[(position - 1) * num_bits // 8]) >> (8 - ((position - 1) * num_bits % 8 + num_bits))) & ((1 << num_bits) - 1)
        position += 1

    return newImage, num_bits, position - 1

def get_message(image):
    message = []
    # Read the number of bits used from the first pixel
    num_bits = image[0] & 0b00000111

    message_bits = []
    position = 1

    while True:
        bits = image[position] & ((1 << num_bits) - 1)
        message_bits.append(bits)
        
        position += 1

        if len(message_bits) * num_bits >= 8:
            char_bits = 0
            for i in range(8 // num_bits):
                char_bits |= (message_bits.pop(0) << (8 - (i + 1) * num_bits))
            if char_bits == 0:
                break
            message.append(chr(char_bits))

        if position >= len(image):
            break

    return ''.join(message)