from ultralytics import YOLO



if __name__ == '__main__':
    #your code
    STV_MNetPath=r'E:\Suyingcai\STV_MNet'
    # 加载模型
    model = YOLO(STV_MNetPath+r'\code\STV_MNet\ultralytics_miou_attention\runs\segment\split_1_datasetODConv_CIoU\weights\best.pt')  # 加载自定义模型
    # # 验证模型
    metrics = model.val(data='data/data.yaml',split='test',conf=0.001)  # 无需参数，数据集和设置记忆
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # 包含每个类别的map50-95列表




 