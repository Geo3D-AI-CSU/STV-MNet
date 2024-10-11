import math
import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
ROOT=r"E:\Suyingcai\STV_MNet"
input_csv_folder=ROOT+r"\data\input data\Location\LNinput_csv"
input_npy_folder=ROOT+r"\results\Location calculation\monoDepth\changsha_Monodepth\output_npy"
output_csv_folder=ROOT+ r'\results\Location calculation\output_csv'

def caldepth(D):
    #焦距F为0.54米
    F=0.54
    #W0设置为721像素
    W0=721
    #W1设置为6656像素
    W1=6656
    #转换参数C设置为1.5
    C=1.5
    return (W0*F*C)/(W1*D)

def calangle(a):
    #图像的宽度的1/2
    w=64
    #sin角度'的值
    angle0=math.sqrt(2)/2
    angle1=a*angle0/w
    #计算弧度
    angle1_radians=math.asin(angle1)
    print("sin(angle1)",math.sin(angle1_radians))
    #弧度转为度数
    angle1_degrees=math.degrees(angle1_radians)
    #45度角
    anglesta=math.radians(45)
    result=anglesta-angle1_radians
    #最终结果
    Result=math.degrees(result)
    return Result,math.sin(angle1_radians)

#角度+45°
def calangle1(a):
    #图像的宽度的1/2
    w=64
    #sin角度'的值
    angle0=math.sqrt(2)/2
    angle1=a*angle0/w
    #计算弧度
    angle1_radians=math.asin(angle1)
    #弧度转为度数
    angle1_degrees=math.degrees(angle1_radians)
    #45度角
    anglesta=math.radians(45)
    result=anglesta+angle1_radians
    #最终结果
    Result=math.degrees(result)
    return Result


#记录最开始的时间
tic=time.time()

def callocation(input_csv_folder,input_npy_folder,output_csv_folder):

    # 获取输入文件夹中的所有CSV文件名
    csv_files = [file for file in os.listdir(input_csv_folder) if file.endswith('.csv')]

    for i,csv_file in enumerate(csv_files):
        if i%10 == 0:
            print (i/10, '    ',"花费时间:",'%.2f'%(time.time()-tic))
        # 构建对应的Numpy文件名
        npy_file = os.path.splitext(csv_file)[0] + "_disp.npy"

        # 使用 split() 方法提取出x和y坐标
        coords = csv_file.split('_')[1].split(',')
        x = float(coords[0])
        y = float(coords[1])

        # 构建输入文件的完整路径
        input_csv_path = os.path.join(input_csv_folder, csv_file)
        input_npy_path = os.path.join(input_npy_folder, npy_file)

        # 构建输出文件的完整路径
        output_csv_path = os.path.join(output_csv_folder, csv_file)

        # 处理单个文件
        process_single_file(input_csv_path, input_npy_path, output_csv_path,x,y)
        print("计算完第",i+1,"个图像")
    

def process_single_file(input_csv_path, input_npy_path, output_csv_path,x,y):
    # 加载CSV文件和Numpy文件的数据
    df = pd.read_csv(input_csv_path)
    data = np.load(input_npy_path)
    #定义数组mylist存放视差值
    my_list=[]

    #定义数组x1和y1保存每个树的xy的偏移量
    x1=[]
    y1=[]
    #z保存树木编号
    # z=[]

    #定义数组dis记录每棵树到图像中线的距离
    dis=[]

    #定义结果坐标的xy坐标
    resultx=[]
    resulty=[]
    resultdepth=[]
    resultangles=[]
    resultsangle1=[]

    for i in range(len(df)):
        data1 = float(df.iloc[i, 1])
        data2 = float(df.iloc[i, 2])
        data3 = float(df.iloc[i, 3])/4
        middle=(data1+data2)/2
        middle=middle/4
        #x的中线值
        middlex=int(middle+0.5)
        #print("第"+str(i)+"颗树的中线值：")
        #print(middlex)
        #y值
        data3=int(data3+0.5)
        if data3>=len(data):
            data3=len(data)-1
        #print("第"+str(i)+"颗树的横坐标：")
        #print(data3)
        my_list.append(data[data3][middlex])
        print("第"+str(i)+"颗树的视差值：")
        print(data[data3][middlex])
        #树的编号
        # z_value=int(df.iloc[i,5])
        # z.append(z_value)
        #计算深度值
        depth=caldepth(my_list[i])
        print("第"+str(i)+"颗树的绝对深度：")
        print(depth)
        resultdepth.append(depth)
        #判断该点的图像位置并计算与图像中线的距离
        if middle>=0 and middle<=64:
            distance=middle
        elif middle>64 and middle<=192:
            distance=abs(middle-128)
        elif middle>192 and middle<=320:
            distance=abs(middle-256)
        elif middle>320 and middle<=448:
            distance=abs(middle-384)
        elif middle>448 and middle<=512:
            distance=abs(middle-512)
        dis.append(distance)
        #print("第"+str(i)+"颗树的distance：")
        #print(distance)
        #计算角度
        resultangle,angle1=calangle(distance)
        resultangles.append(resultangle)
        resultsangle1.append(angle1)
        #print("第"+str(i)+"颗树的角度：")
        #print(resultangle)
        resultangle_radians=math.radians(resultangle)
        sin_value=math.sin(resultangle_radians)
        cos_value=math.cos(resultangle_radians)
        #print("第"+str(i)+"颗树的cos值：")
        #print(cos_value)
        #print("第"+str(i)+"颗树的sin值：")
        #print(sin_value)
        xD=depth*cos_value/100000
        yD=depth*sin_value/100000
        #print("第"+str(i)+"颗树的x增加量：")
        #print(xD)
        #print("第"+str(i)+"颗树的y增加量：")
        #print(yD)
        x1.append(xD)
        y1.append(yD)
        #原来的部分
        #resultx.append(x+xD)
        #resulty.append(y+yD)
        #重新修改的部分
        if(middle>=256):
            resultx.append(x+xD)
            resulty.append(y+yD)
        elif(middle<=128):
            resultx.append(x-xD)
            resulty.append(y-yD)
        else :
            resultx.append(x-xD)
            resulty.append(y+yD)

       
    output_csv_path = os.path.join(output_csv_folder,  'pro_'+os.path.basename(output_csv_path))
    # 将坐标数据写入CSV文件
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID','left','top','right','bottom','depth','sin(angle1)','x', 'y'])  # 写入表头
        for i in range(len(resultx)):
            writer.writerow([i,df.iloc[i, 1],df.iloc[i, 3],df.iloc[i, 2],df.iloc[i, 4],resultdepth[i],resultsangle1[i],resultx[i], resulty[i]])  # 逐行写入坐标数据


def main():
    #调用计算函数
    callocation(input_csv_folder,input_npy_folder,output_csv_folder)

if __name__ == "__main__":
    main()
