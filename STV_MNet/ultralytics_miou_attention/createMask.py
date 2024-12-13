import os
import json
from PIL import Image, ImageDraw

def generate_rgb_mask(json_path, output_dir, image_width=2048, image_height=1024):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mask = Image.new('RGB', (image_width, image_height), (0, 0, 0))
    draw = ImageDraw.Draw(mask)

    for segment in data:
        points = list(zip(segment["segments"]["x"], segment["segments"]["y"]))
        if len(points )!=0:
            draw.polygon(points, fill=(128, 0, 0))

    mask_filename = os.path.splitext(os.path.basename(json_path))[0] + '.png'
    mask_path = os.path.join(output_dir, mask_filename)
    mask.save(mask_path)

def generate_rgb_masks_from_json_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(input_dir, filename)
            generate_rgb_mask(json_path, output_dir)


STV_MNetPath=r'E:\Suyingcai\STV_MNet'
input_dir = STV_MNetPath+r'\results\STV_MNet\predict_changsha\json_seg'
output_dir = STV_MNetPath+r'\data\input data\Structure\test'

generate_rgb_masks_from_json_files(input_dir, output_dir)
print("OK")

