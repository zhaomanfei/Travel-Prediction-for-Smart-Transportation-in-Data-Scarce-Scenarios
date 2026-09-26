import xml.etree.ElementTree as ET
import csv

# 输入XML文件（SUMO生成的FCD数据）
XML_FILE = "fcd_data.xml"
# 输出CSV文件
CSV_FILE = "fcd_data.csv"

# 解析XML
tree = ET.parse(XML_FILE)
root = tree.getroot()

# 准备CSV写入器
with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # 写入表头
    writer.writerow(["time", "id", "x", "y", "speed", "edge_id"])
    
    # 遍历每一秒的车辆数据
    for timestep in root.findall("timestep"):
        time = timestep.get("time")
        # 遍历该时间步的所有车辆
        for vehicle in timestep.findall("vehicle"):
            vehicle_id = vehicle.get("id")
            x = vehicle.get("x")
            y = vehicle.get("y")
            speed = vehicle.get("speed")
            edge_id = vehicle.get("lane").split("_")[0]  # 从车道ID中提取路段ID
            # 写入一行数据
            writer.writerow([time, vehicle_id, x, y, speed, edge_id])

print(f"成功将{XML_FILE}转换为{CSV_FILE}")