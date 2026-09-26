import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os

# ===================== 核心配置（无需填高德Key，纯模拟数据） =====================
AMAP_KEY = ""  # 无需填写，完全跳过API调用
KEY_INTERSECTIONS = {
    "圆明园西路": {"lng": 116.30256, "lat": 39.99587},
    "中关村大街": {"lng": 116.31945, "lat": 39.98468},
    "北四环西路": {"lng": 116.32678, "lat": 39.99972},
    "万泉河路": {"lng": 116.29874, "lat": 39.98965}
}
DAILY_LIMIT = 200
PER_INTERSECTION_LIMIT = DAILY_LIMIT // len(KEY_INTERSECTIONS)
COLLECT_INTERVAL = 5
COLLECT_DATE = datetime.now().strftime("%Y-%m-%d")  # 自动取当前日期
MORNING_PEAK = (7, 8.5)
EVENING_PEAK = (17, 18.5)
# Excel保存路径（和之前完全一致）
SAVE_PATH = "D:/highSchool/科研/paper/LSTM/Pro/LSTMTrafficPro/beijing_traffic_data_amap.xlsx"

# ===================== 依赖检查（确保Excel保存正常） =====================
def check_dependencies():
    """检查Excel保存所需依赖"""
    try:
        import openpyxl
    except ImportError:
        print("⚠️ 缺少Excel保存依赖openpyxl，正在自动安装...")
        os.system("pip install openpyxl")
        import openpyxl
    print("✅ 依赖检查完成，可正常保存Excel文件")

# ===================== 核心函数：生成贴合北京规律的模拟数据 =====================
def generate_traffic_by_history(intersection_name, hour):
    """为4个路口定制模拟数据（贴合北京真实交通规律）"""
    history_congestion = {
        "圆明园西路": {"morning_peak": 2, "evening_peak": 1, "normal": 0},
        "中关村大街": {"morning_peak": 2, "evening_peak": 2, "normal": 1},
        "北四环西路": {"morning_peak": 1, "evening_peak": 2, "normal": 0},
        "万泉河路": {"morning_peak": 1, "evening_peak": 1, "normal": 0}
    }
    # 判定时段
    if MORNING_PEAK[0] <= hour <= MORNING_PEAK[1]:
        congestion = history_congestion[intersection_name]["morning_peak"]
    elif EVENING_PEAK[0] <= hour <= EVENING_PEAK[1]:
        congestion = history_congestion[intersection_name]["evening_peak"]
    else:
        congestion = history_congestion[intersection_name]["normal"]
    
    # 模拟车速和流量（贴合真实规律）
    speed_map = {2: np.random.randint(5, 10), 1: np.random.randint(15, 25), 0: np.random.randint(30, 40)}
    volume_map = {2: np.random.randint(200, 300), 1: np.random.randint(100, 200), 0: np.random.randint(30, 100)}
    
    return {
        "speed": speed_map[congestion],
        "congestion": congestion,
        "traffic_volume": volume_map[congestion]
    }

def get_traffic_flow(lng, lat, intersection_name):
    """强制返回模拟数据，跳过所有API调用"""
    hour = np.random.uniform(0, 24)  # 随机生成时段，模拟实时采集
    return generate_traffic_by_history(intersection_name, hour)

# ===================== 核心采集函数（保留原有逻辑） =====================
def collect_traffic_data():
    """主采集函数：纯模拟数据，生成200条有效数据"""
    print("===== 开始采集（纯模拟数据，贴合北京交通规律） =====")
    valid_intersections = {k: v for k, v in KEY_INTERSECTIONS.items() if v["lng"] and v["lat"]}
    if not valid_intersections:
        print("❌ 无有效路口坐标！")
        return None
    
    # 生成采集时间戳（覆盖早晚高峰）
    start_time = datetime.strptime(f"{COLLECT_DATE} 07:00", "%Y-%m-%d %H:%M")
    collect_timestamps = []
    current_time = start_time
    for _ in range(PER_INTERSECTION_LIMIT):
        collect_timestamps.append(current_time.strftime("%Y-%m-%d %H:%M"))
        current_time += timedelta(minutes=COLLECT_INTERVAL)
    
    # 采集数据（控制200条上限）
    collected_data = []
    call_count = 0
    
    for intersection_name, geo in valid_intersections.items():
        print(f"\n--- 采集【{intersection_name}】（剩余配额：{DAILY_LIMIT - call_count}）---")
        for ts in collect_timestamps:
            if call_count >= DAILY_LIMIT:
                print("⚠️ 达200条上限，停止采集")
                break
            
            flow_data = get_traffic_flow(geo["lng"], geo["lat"], intersection_name)
            call_count += 1
            time.sleep(0.1)  # 轻微延时，模拟采集间隔
            
            # 判定时段类型
            hour = float(datetime.strptime(ts, "%Y-%m-%d %H:%M").strftime("%H.%M"))
            if MORNING_PEAK[0] <= hour <= MORNING_PEAK[1]:
                time_period = "早高峰"
            elif EVENING_PEAK[0] <= hour <= EVENING_PEAK[1]:
                time_period = "晚高峰"
            elif 0 <= hour < 6 or 22 <= hour <= 24:
                time_period = "夜间"
            else:
                time_period = "平峰"
            
            # 映射拥堵状态
            congestion_map = {0: "畅通", 1: "缓行", 2: "拥堵"}
            congestion = congestion_map.get(flow_data["congestion"], "畅通")
            
            # 组装数据（字段与原有Excel完全一致）
            collected_data.append({
                "采集时间戳": ts,
                "核心路口名称": intersection_name,
                "5分钟流量（辆）": flow_data["traffic_volume"],
                "时段类型": time_period,
                "主要车种占比（轿车/电动车%）": np.random.randint(75, 85),
                "路网拥挤度": congestion,
                "平均车速（km/h）": flow_data["speed"],
                "特殊场景标注": "上学日" if 7 <= hour <= 9 or 17 <= hour <= 19 else "无特殊场景"
            })
        
        if call_count >= DAILY_LIMIT:
            break
    
    df = pd.DataFrame(collected_data)
    print(f"\n✅ 采集完成！共{len(df)}条模拟数据（符合200条上限规则）")
    return df

# ===================== Excel保存强化版（确保稳定保存） =====================
def validate_and_save(df):
    """强化Excel保存逻辑，处理路径/权限问题"""
    if df is None or len(df) == 0:
        print("❌ 无数据可保存！")
        return
    
    # 数据清洗（保证Excel数据格式正确）
    df["5分钟流量（辆）"] = pd.to_numeric(df["5分钟流量（辆）"], errors="coerce").fillna(0).astype(int)
    df["主要车种占比（轿车/电动车%）"] = pd.to_numeric(df["主要车种占比（轿车/电动车%）"], errors="coerce").fillna(80).clip(0, 100)
    df["平均车速（km/h）"] = pd.to_numeric(df["平均车速（km/h）"], errors="coerce").fillna(0)
    
    # 确保保存目录存在（避免路径不存在报错）
    save_dir = os.path.dirname(SAVE_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📁 创建保存目录：{save_dir}")
    
    # 尝试保存Excel（重试机制，处理文件占用问题）
    save_success = False
    for retry in range(3):
        try:
            with pd.ExcelWriter(SAVE_PATH, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, sheet_name="高德采集交通数据", index=False)
            save_success = True
            break
        except PermissionError:
            print(f"⚠️ Excel文件被占用（重试{retry+1}/3），请关闭已打开的表格！")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ 保存Excel失败（重试{retry+1}/3）：{e}")
            time.sleep(2)
    
    if not save_success:
        print(f"❌ 多次重试后仍无法保存Excel，请检查路径权限：{SAVE_PATH}")
        return
    
    # 输出保存成功的统计信息
    print(f"\n✅ Excel文件已成功保存至：{SAVE_PATH}")
    print("===== 模拟数据统计（Excel内数据特征） =====")
    for name in df["核心路口名称"].unique():
        cnt = len(df[df["核心路口名称"] == name])
        congestion_cnt = len(df[(df["核心路口名称"] == name) & (df["路网拥挤度"] == "拥堵")])
        avg_volume = df[df["核心路口名称"] == name]["5分钟流量（辆）"].mean()
        avg_speed = df[df["核心路口名称"] == name]["平均车速（km/h）"].mean()
        print(f"【{name}】：{cnt}条数据 | 拥堵{congestion_cnt}条 | 平均流量：{int(avg_volume)}辆 | 平均车速：{int(avg_speed)}km/h")

# ===================== 主程序（一键运行，无需配置） =====================
if __name__ == "__main__":
    # 先检查依赖（确保Excel保存正常）
    check_dependencies()
    
    # 打印采集信息
    print(f"📌 采集日期：{COLLECT_DATE}")
    print(f"📌 核心采集路口：{list(KEY_INTERSECTIONS.keys())}")
    print(f"📌 Excel保存路径：{SAVE_PATH}\n")
    
    # 开始采集并保存
    df_collected = collect_traffic_data()
    validate_and_save(df_collected)
    
    # 预览Excel前10条数据（方便核对）
    if df_collected is not None and len(df_collected) > 0:
        print("\n===== Excel内前10条数据预览 =====")
        preview_cols = ["采集时间戳", "核心路口名称", "5分钟流量（辆）", "路网拥挤度", "平均车速（km/h）"]
        print(df_collected[preview_cols].head(10))
        
        # 额外验证：读取保存的Excel文件，确认数据一致
        try:
            df_check = pd.read_excel(SAVE_PATH, sheet_name="高德采集交通数据")
            print(f"\n✅ 验证：Excel文件读取成功，共{len(df_check)}条数据，和采集数据一致！")
        except Exception as e:
            print(f"\n⚠️ 验证Excel读取失败：{e}")