from PIL import ImageFont, ImageDraw, Image
from io import BytesIO

image = Image.new('RGB',(128,64))
draw = ImageDraw.Draw(image)

file = open("arial.ttf", "rb")
bytes_font = BytesIO(file.read())
font = ImageFont.truetype(bytes_font, 15)

# use a bitmap font
#font = ImageFont.load("arial.pil")
#font = ImageFont.truetype("arial.ttf", 15)

#draw.text((10, 10), "hello", font=font)

# use a truetype font
#font = ImageFont.truetype("arial.ttf", 15)

draw.text((10, 25), "world", font=font)
