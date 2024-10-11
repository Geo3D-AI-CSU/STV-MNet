STV-MNet代码程序：
服务器环境：yolov8

代码文件说明：
labelme2txt.py为labelme标注文件转yolo v8训练数据格式的代码
train_seg.py为训练代码
val.py为输出验证集、测试集指标得分的代码
test.py为预测代码
createMask.py为yolo v8使用预测结果生成mask图，作为结构测算输入的代码
STV_MNet\code\STV_MNet\ultralytics_miou_attention\ultralytics\cfg\models\v8\yolov8-seg-ODConv_3rd.yaml为改进的网络模型配置文件
YOLO v8添加ODConv注意力机制博客：https://blog.csdn.net/qq_69854365/article/details/132840135
WIoU修改参考博客：https://blog.csdn.net/qq_46542320/article/details/135057759



###############################################################
YOLO v3代码：
服务器环境：yolov3_ultralytics


pixel2box为将语义分割json标签数据转为目标检测json标签数据
json_visualization.py为检查json标注数据是否转换正确
bboxJson2txt.py为将目标检测json标签数据转为yolo v3训练txt格式数据
Deeplab_YOLO_TXT.py为将Deeplab之前划分好的训练集、验证集、测试集划为对应的YOLO训练的训练集、验证集、测试集列表
makeTXT.py为将数据集列表将对应的txt文件放入train、val、test位置中
train为yolo v3训练代码
val.py为输出验证集、测试集指标得分的代码
detect.py为预测代码

#######################
Deeplab v3代码：
服务器环境：deeplabv3
详细见项目中的readme





注：各网络data/name.yaml文件为个网络训练集、验证集、测试集地址，需按项目进行修改
