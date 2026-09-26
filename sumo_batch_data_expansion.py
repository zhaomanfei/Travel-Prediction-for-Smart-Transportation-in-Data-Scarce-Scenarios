import os
import subprocess
import xml.etree.ElementTree as ET
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm

# ===================== 配置参数（需根据你的工程路径修改） =====================
# 1. 基础路径配置
SUMO_BIN_PATH = "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo.exe"  # 若sumo未添加到环境变量，需填写完整路径（如"D:\SUMO\bin\sumo.exe"）
SUMO_CFG_FILE = "osm_zgc.sumocfg"  # 你的SUMO配置文件名称
ROUTES_FILE = "routes.xml"  # 你的车流配置文件名称
NET_FILE = "osm.zgc.xml"  # 你的路网文件名称

# 2. 批量仿真配置
SIMULATION_TIMES = 10  # 批量运行仿真次数（10倍扩容）
SIMULATION_DURATION = 3600  # 单次仿真时长（秒），建议3600秒（1小时）提升数据量
DELAY_MS = 100  # 仿真延迟（仅GUI有效，命令行运行无影响）

# 3. 数据输出配置
TEMP_XML_PREFIX = "temp_fcd_data_"  # 临时FCD XML文件前缀
TEMP_CSV_PREFIX = "temp_fcd_data_"  # 临时FCD CSV文件前缀
FINAL_FCD_CSV = "fcd_data_big.csv"  # 合并后的最终FCD CSV文件
FINAL_TRAFFIC_CSV = "traffic_road_level_big.csv"  # 最终路段级扩容数据文件
SEQ_LEN = 12  # LSTM历史时间步（参考PEMS04）
PRED_LEN = 3  # LSTM预测时间步（参考PEMS04）

# ===================== 工具函数：XML转CSV（无依赖SUMO的xml2csv.py） =====================
def xml_to_csv(xml_file, csv_file):
    """
    将SUMO生成的FCD XML文件转换为CSV文件
    :param xml_file: 输入FCD XML路径
    :param csv_file: 输出CSV路径
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"解析XML文件失败 {xml_file}：{e}")
        return False

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 写入CSV表头
        writer.writerow(["time", "id", "x", "y", "speed", "edge_id"])

        # 遍历每个时间步的车辆数据
        for timestep in root.findall("timestep"):
            time = timestep.get("time")
            if not time:
                continue
            # 遍历该时间步下的所有车辆
            for vehicle in timestep.findall("vehicle"):
                vehicle_id = vehicle.get("id", "")
                x = vehicle.get("x", "")
                y = vehicle.get("y", "")
                speed = vehicle.get("speed", "")
                lane = vehicle.get("lane", "")
                # 从车道ID中提取路段ID（车道ID格式：edge_id_车道号）
                edge_id = lane.split("_")[0] if "_" in lane else lane

                # 写入一行数据
                writer.writerow([time, vehicle_id, x, y, speed, edge_id])
    return True

# ===================== 步骤1：批量运行SUMO仿真，生成FCD XML数据 =====================
def batch_sumo_simulation():
    """批量运行SUMO仿真，生成10个FCD XML临时文件"""
    print("=" * 50)
    print("开始批量运行SUMO仿真（共{}次）...".format(SIMULATION_TIMES))
    print("=" * 50)

    # 先检查核心配置文件是否存在
    for cfg_file in [SUMO_CFG_FILE, ROUTES_FILE, NET_FILE]:
        if not os.path.exists(cfg_file):
            raise FileNotFoundError(f"核心配置文件缺失：{cfg_file}，请确认文件路径正确")

    # 循环运行仿真
    for i in tqdm(range(1, SIMULATION_TIMES + 1), desc="仿真进度"):
        # 定义本次仿真的FCD XML输出路径
        temp_xml = f"{TEMP_XML_PREFIX}{i}.xml"
        
        # 构建SUMO运行命令（移除无效的--duration参数，保留核心有效参数）
        cmd = [
            SUMO_BIN_PATH,
            "-c", SUMO_CFG_FILE,
            "--fcd-output", temp_xml,  # 指定本次仿真的FCD输出文件
            "--no-step-log",  # 关闭步骤日志，提升运行速度
            "--no-warnings"  # 关闭非关键警告，减少输出干扰
        ]

        try:
            # 执行仿真命令（开启日志输出，方便排查问题）
            subprocess.run(
                cmd,
                encoding="utf-8",
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"第{i}次仿真运行失败：{e.stderr}")
            continue
        except FileNotFoundError:
            raise FileNotFoundError(f"SUMO可执行文件未找到，请检查SUMO_BIN_PATH配置是否正确")

    print("批量仿真完成，生成{}个FCD XML文件".format(SIMULATION_TIMES))

# ===================== 步骤2：批量转换XML到CSV，解析FCD数据 =====================
def batch_xml_to_csv():
    """批量将FCD XML文件转换为CSV文件"""
    print("=" * 50)
    print("开始批量转换XML到CSV...")
    print("=" * 50)

    csv_files = []
    for i in tqdm(range(1, SIMULATION_TIMES + 1), desc="XML转CSV进度"):
        temp_xml = f"{TEMP_XML_PREFIX}{i}.xml"
        temp_csv = f"{TEMP_CSV_PREFIX}{i}.csv"
        if not os.path.exists(temp_xml):
            print(f"跳过不存在的XML文件：{temp_xml}")
            continue

        # 转换XML到CSV
        if xml_to_csv(temp_xml, temp_csv):
            csv_files.append(temp_csv)
        else:
            print(f"第{i}个文件转换失败：{temp_xml}")

    print(f"XML转CSV完成，成功生成{len(csv_files)}个CSV文件")
    return csv_files

# ===================== 步骤3：合并多个CSV文件，生成超大FCD数据集 =====================
def merge_csv_files(csv_files):
    """合并多个临时FCD CSV文件，生成单一超大CSV文件"""
    print("=" * 50)
    print("开始合并CSV文件...")
    print("=" * 50)

    if not csv_files:
        raise ValueError("无有效CSV文件可供合并")

    # 批量读取CSV文件并合并
    df_list = []
    for csv_file in tqdm(csv_files, desc="CSV合并进度"):
        try:
            df = pd.read_csv(csv_file)
            # 给每个批次的数据添加批次标签，避免时间和车辆ID重复
            df["batch_id"] = os.path.basename(csv_file).replace(TEMP_CSV_PREFIX, "").replace(".csv", "")
            df_list.append(df)
        except Exception as e:
            print(f"读取CSV文件失败 {csv_file}：{e}")
            continue

    # 合并所有DataFrame
    big_df = pd.concat(df_list, ignore_index=True)

    # 数据清洗：去除无效列和空值
    big_df = big_df[["time", "id", "batch_id", "edge_id", "speed"]].dropna()
    # 转换数据类型，确保后续处理正常
    big_df["time"] = pd.to_numeric(big_df["time"], errors="coerce").fillna(0).astype(int)
    big_df["speed"] = pd.to_numeric(big_df["speed"], errors="coerce").fillna(0)
    big_df["edge_id"] = big_df["edge_id"].astype(str)
    # 重新生成唯一时间戳（避免批次间时间重复，实现时序连续）
    big_df["global_time"] = big_df["time"] + (big_df["batch_id"].astype(int) - 1) * SIMULATION_DURATION

    # 保存合并后的超大CSV文件
    big_df.to_csv(FINAL_FCD_CSV, index=False)
    print(f"CSV文件合并完成，生成超大FCD数据文件：{FINAL_FCD_CSV}")
    print(f"合并后数据总量：{len(big_df)} 条记录")
    return big_df

# ===================== 步骤4：插值补全，生成路段级时序数据 =====================
def traffic_data_aggregation_and_interpolation(big_df):
    """从合并后的FCD数据中聚合路段级特征，并进行时序插值补全"""
    print("=" * 50)
    print("开始聚合路段级特征并插值补全...")
    print("=" * 50)

    # 1. 聚合为路段级时序数据（时间步+路段ID → 流量、平均速度）
    traffic_df = big_df.groupby(["global_time", "edge_id"]).agg(
        flow=("speed", "count"),  # 流量=该路段该时间的车辆数
        avg_speed=("speed", "mean")  # 平均速度=该路段该时间所有车辆的速度均值
    ).reset_index()

    # 2. 数据重排：转换为「时间步×路段ID」的透视表（对齐PEMS04格式）
    traffic_pivot = traffic_df.pivot(
        index="global_time",
        columns="edge_id",
        values=["flow", "avg_speed"]
    )

    # 3. 填充空值并插值补全
    # 先填充0值（无车辆的路段），再进行线性插值（补全稀疏时序）
    traffic_pivot = traffic_pivot.fillna(0)
    traffic_pivot = traffic_pivot.interpolate(
        method="linear",
        limit_direction="both",  # 向前+向后插值，确保首尾数据完整
        axis=0  # 按时间轴插值
    )

    # 4. 还原为原始表格格式
    traffic_flat = traffic_pivot.reset_index()
    traffic_flat.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in traffic_flat.columns]
    traffic_flat.rename(columns={"global_time_": "global_time"}, inplace=True)

    # 5. 重塑为LSTM适配的格式（时间步、路段ID、流量、平均速度）
    final_traffic = []
    for idx, row in traffic_flat.iterrows():
        time_step = row["global_time"]
        for edge_id in traffic_df["edge_id"].unique():
            flow_col = f"flow_{edge_id}"
            speed_col = f"avg_speed_{edge_id}"
            flow = row[flow_col] if flow_col in traffic_flat.columns else 0
            avg_speed = row[speed_col] if speed_col in traffic_flat.columns else 0
            final_traffic.append({
                "time": time_step,
                "edge_id": edge_id,
                "flow": flow,
                "avg_speed": avg_speed
            })

    # 转换为DataFrame并保存
    final_traffic_df = pd.DataFrame(final_traffic)
    final_traffic_df = final_traffic_df.sort_values(by=["time", "edge_id"]).reset_index(drop=True)
    final_traffic_df.to_csv(FINAL_TRAFFIC_CSV, index=False)

    # 输出数据统计信息
    total_time_steps = final_traffic_df["time"].nunique()
    total_edges = final_traffic_df["edge_id"].nunique()
    total_samples = len(final_traffic_df)
    print("数据插值补全完成，生成最终路段级数据文件：{}".format(FINAL_TRAFFIC_CSV))
    print(f"数据统计：")
    print(f"  - 总时间步：{total_time_steps}（约{total_time_steps//3600}小时）")
    print(f"  - 覆盖路段数：{total_edges}")
    print(f"  - 总数据样本数：{total_samples}（10倍扩容完成）")

# ===================== 步骤5：清理临时文件（可选） =====================
def clean_temp_files():
    """清理批量仿真生成的临时XML和CSV文件"""
    print("=" * 50)
    print("开始清理临时文件...")
    print("=" * 50)

    for i in range(1, SIMULATION_TIMES + 1):
        temp_xml = f"{TEMP_XML_PREFIX}{i}.xml"
        temp_csv = f"{TEMP_CSV_PREFIX}{i}.csv"
        for file in [temp_xml, temp_csv]:
            if os.path.exists(file):
                os.remove(file)
    print("临时文件清理完成")

# ===================== 主函数：一键执行全流程 =====================
if __name__ == "__main__":
    try:
        # 步骤1：批量SUMO仿真
        batch_sumo_simulation()

        # 步骤2：批量XML转CSV
        csv_files = batch_xml_to_csv()

        # 步骤3：合并CSV文件
        big_fcd_df = merge_csv_files(csv_files)

        # 步骤4：聚合+插值补全
        traffic_data_aggregation_and_interpolation(big_fcd_df)

        # 步骤5：清理临时文件（如需保留临时文件，注释该行即可）
        clean_temp_files()

        print("=" * 50)
        print("全流程执行完成！10倍规模路段级数据已生成，可直接用于LSTM模型训练")
        print("核心输出文件：")
        print(f"  1. 合并后FCD轨迹数据：{FINAL_FCD_CSV}")
        print(f"  2. 最终路段级时序数据：{FINAL_TRAFFIC_CSV}")
    except Exception as e:
        print(f"程序执行失败：{e}")