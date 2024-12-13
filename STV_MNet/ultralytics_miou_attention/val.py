from ultralytics import YOLO

if __name__ == '__main__':
    # Define the path to the STV_MNet directory
    STV_MNetPath = r'E:\Suyingcai\STV_MNet'

    # Load the model
    model = YOLO(STV_MNetPath + r'\code\STV_MNet\ultralytics_miou_attention\runs\segment\split_1_datasetODConv_CIoU\weights\best.pt')  # Load the custom model

    # Validate the model
    metrics = model.val(data='data/data.yaml', split='test', conf=0.001)  # No need to specify parameters, dataset and settings are remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # List of map50-95 for each class





 