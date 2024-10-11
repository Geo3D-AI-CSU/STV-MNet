# 将采用Label中多边形工具标注的用于语义分割的原始json文件转换成 用于目标检测的用矩形框标注的原始json文件
import json
import os


def convert_semantic_to_detection(input_file, output_file):
    with open(input_file, 'r',encoding='utf-8') as f:
        data = json.load(f)

    # 构建目标检测的JSON结构
    detection_data = {
        "version": "5.2.0.post4",
        "flags": {},
        "shapes": [],
        "imagePath": data['imagePath'],
        "imageData": None,
        "imageHeight": data['imageHeight'],
        "imageWidth": data['imageWidth'],
    }

    for shape in data["shapes"]:
        # 转换多边形为矩形
        label = shape["label"]
        points = shape["points"]
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x_min = min(x_values)
        y_min = min(y_values)
        x_max = max(x_values)
        y_max = max(y_values)

        # 添加矩形边界框到目标形状列表
        detection_data["shapes"].append({
            "label": label,
            "points": [[x_min, y_min],[x_max,y_min], [x_max, y_max],[x_min, y_max]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {}
        })

    with open(output_file, 'w') as f:
        json.dump(detection_data, f, indent=4)

    print("转换完成！")

Yolo_v3Path=r"E:\Suyingcai\STV_MNet\code\STV_MNet\yolov3-master"
input_path = Yolo_v3Path+r"\data\labelsSegment\pixel"
output_path = Yolo_v3Path+r"\data\labelsSegment\bbox"
for file_name in os.listdir(input_path):
    input_json_path = input_path + "/" + file_name
    print('input_json_path:',input_json_path)
    output_json_path = output_path + "/" + file_name
    print('output_json_path:', output_json_path)

    # 使用示例
    convert_semantic_to_detection(input_json_path, output_json_path)
