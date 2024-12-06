import pyproj
import csv


def convert_coordinates(input_csv, output_csv):
    """
    将CSV文件中的WGS84坐标转换为UTM坐标，并写入新的CSV文件。

    Args:
        input_csv: 输入CSV文件的路径。
        output_csv: 输出CSV文件的路径。
    """

    try:
        wgs84 = pyproj.CRS("EPSG:4326")
        utm = pyproj.CRS("EPSG:32650")
        transformer = pyproj.Transformer.from_crs(wgs84, utm)

        with open(input_csv, 'r', encoding='utf-8') as infile, open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 写入表头 (假设输入CSV文件的第一行为表头)
            header = next(reader)
            writer.writerow(header + ["UTM_X", "UTM_Y"]) # 添加UTM坐标列
            i=1
            for row in reader:
                try:
                    lon = float(row[0])
                    lat = float(row[1])
                    x, y = transformer.transform(lat, lon)  #注意lat,lon的顺序
                    print(i)
                    i=i+1
                    writer.writerow(row + [x, y])
                except (ValueError, IndexError) as e:
                    print(f"错误:  处理行 {row} 时出错: {e}")
                    #可以选择跳过错误行或者采取其他错误处理方式
                    # writer.writerow(row + ["Error", "Error"]) #例如写入错误标记

    except FileNotFoundError:
        print(f"错误: 输入文件 {input_csv} 不存在。")
    except Exception as e:
        print(f"错误: 发生未知错误: {e}")


# 使用示例
input_file = r"C:\data\new_structure\xy_wgs84.csv"  # 替换为你的输入CSV文件路径
output_file = r"C:\data\new_structure\xy_utm.csv" # 替换为你的输出CSV文件路径
convert_coordinates(input_file, output_file)
print("ok")