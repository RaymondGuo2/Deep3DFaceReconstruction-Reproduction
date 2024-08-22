from PIL import Image, ExifTags
import os


def process_orientation(image, orientation):
    if orientation == 1:
        return image
    elif orientation == 2:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        return image.rotate(180, expand=True)
    elif orientation == 4:
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    elif orientation == 5:
        return image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 6:
        return image.rotate(90, expand=True)
    elif orientation == 7:
        return image.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM)
    elif orientation == 8:
        return image.rotate(-90, expand=True)
    else:
        return image




def remove_exif(path):
    image = Image.open(path)
    exif_data = image.getexif()
    if exif_data:
        print(f"Detected exif_data in {os.path.basename(path)}")
        orientation = exif_data.get(274)
        image = process_orientation(image, orientation)
        
        data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(data)
        image_without_exif.save(path)
        # rotated_image = image_without_exif.rotate(-90, expand=True)
        # rotated_image.save(path)
        image.close()
        image_without_exif.close()
        # rotated_image.close()
        return 
    else:
        print(f"Image not affected - did not detect exif data in {os.path.basename(path)}")
        return 







