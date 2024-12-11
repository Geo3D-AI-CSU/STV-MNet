STV-MNet Code Program:
Server Environment: YOLOv8
Code File Descriptions:

labelme2txt.py: Converts LabelMe annotation files to YOLOv8 training data format.
train_seg.py: The training script.
val.py: Outputs validation and test set metric scores.
test.py: The prediction script.
createMask.py: Generates mask images from YOLOv8 prediction results, used as input for structural calculations.
STV_MNet\code\STV_MNet\ultralytics_miou_attention\ultralytics\cfg\models\v8\yolov8-seg-ODConv_3rd.yaml: The configuration file for the improved network model.
Blog on adding ODConv attention mechanism to YOLO v8: https://blog.csdn.net/qq_69854365/article/details/132840135
Blog on WIoU modification reference: https://blog.csdn.net/qq_46542320/article/details/135057759


YOLOv3 Code:
Server Environment: YOLOv3 (Ultralytics)
Code File Descriptions:

pixel2box: Converts semantic segmentation JSON labels to object detection JSON labels.
json_visualization.py: Used to check if JSON label data is converted correctly.
bboxJson2txt.py: Converts object detection JSON labels to YOLOv3 training TXT format data.
Deeplab_YOLO_TXT.py: Converts the training, validation, and test sets pre-split by Deeplab into the corresponding YOLO training set, validation set, and test set lists.
makeTXT.py: Places corresponding TXT files into the train, val, and test directories based on the dataset list.
train: The YOLOv3 training script.
val.py: Outputs validation and test set metric scores.
detect.py: The prediction script.


Deeplab v3 Code:
Server Environment: Deeplabv3
For detailed instructions, refer to the README in the project.


Note:
Each network’s data/name.yaml file contains the paths to the training, validation, and test datasets for that network. These paths need to be updated according to the specific project.
