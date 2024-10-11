

from ultralytics import YOLO
from pathlib import Path
 

if __name__ == '__main__':
    STV_MNetPath=r'E:\Suyingcai\STV_MNet'
    trainPath=STV_MNetPath+r'\data\input data\STV_MNet\data\trainval\txt'

    ds_yamls = [trainPath+'/2024-03-20_5-Fold_Cross-Valid/split_1/split_1_dataset.yaml', 
     trainPath+'/2024-03-20_5-Fold_Cross-Valid/split_2/split_2_dataset.yaml', 
     trainPath+'/2024-03-20_5-Fold_Cross-Valid/split_3/split_3_dataset.yaml',
     trainPath+'/2024-03-20_5-Fold_Cross-Valid/split_4/split_4_dataset.yaml',
     trainPath+'/2024-03-20_5-Fold_Cross-Valid/split_5/split_5_dataset.yaml']
    
    # model = YOLO('yolov8n.pt', task='segment')
    modelpath = 'weights/yolov8x-seg.pt'
    # modelpath = 'runs/segment/split_3_dataset/weights/last.pt'

    results = {}
    for k in range(5):
        dataset_yaml = ds_yamls[k]
        name = Path(dataset_yaml).stem
        # model = YOLO(modelpath,task='segment')  # load a pretrained model (recommended for training)
        model = YOLO('yolov8-seg-ODConv_3rd.yaml',task='segment').load(modelpath)  # load a pretrained model (recommended for training)
        # Train the model
        yamlpath = 'data/data.yaml'
        model.train(epochs=5,data=dataset_yaml,imgsz=[1024,512],name=name+'ODConv_3rd_WIoU_x')
        results[k] = model.metrics  # save output metrics for further analysis


    print("*"*40)
    print("K-Fold Cross Validation Completed.")
    print("*"*40)
    