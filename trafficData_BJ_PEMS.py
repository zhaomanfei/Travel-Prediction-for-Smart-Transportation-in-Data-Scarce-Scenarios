import requests
import csv
import time
import os
import json
from datetime import datetime
import numpy as np
import random

# -------------------------- 配置参数 --------------------------
GAODE_AK = "替换为新创建的Web服务AK"  # 必须是Web服务AK，且开通交通态势API
CITY_CODE = "110000"  # 北京城市编码
RECTANGLE_RANGE = "116.2800,39.9600;116.3300,39.9900"  # 缩小的矩形区域
INTERVAL = 300  # 采集间隔（5分钟）
MAX_COLLECT_COUNT = 288  # 单日最大采集次数
SAVE_DIR = os.path.join(os.getcwd(), "BJ_PEMS_Traffic_Data")  # 改为当前脚本目录（避免桌面权限问题）
ROAD_COUNT_TARGET = 50  # 目标采集道路数量
USE_MOCK_DATA = True  # API失败时启用模拟数据（临时应急）
# -------------------------------------------------------------

# 初始化道路名称缓存
road_name_cache = []

def init_road_cache():
    """初始化道路名称缓存（本地生成，不依赖API）"""
    global road_name_cache
    road_list_path = os.path.join(SAVE_DIR, "road_list.json")
    road_name_txt_path = os.path.join(SAVE_DIR, "roadname.txt")
    
    # 加载本地缓存
    if os.path.exists(road_list_path):
        try:
            with open(road_list_path, "r", encoding="utf-8") as f:
                road_name_cache = json.load(f)[:ROAD_COUNT_TARGET]
            print(f"加载已保存的道路列表，共{len(road_name_cache)}条道路")
        except:
            road_name_cache = []
    else:
        road_name_cache = []
    
    # 本地生成道路名（圆明园西路周边）
    if len(road_name_cache) == 0:
        #base_roads = [
        #    "圆明园西路", "中关村大街", "北四环西路", "万泉河路", "苏州街",
        #    "海淀南路", "丹棱街", "科学院南路", "知春路", "学院路"

            
        #]

        base_roads = [
            # 圆明园西路核心周边
            "圆明园西路", "中关村大街", "北四环西路", "万泉河路", "苏州街",
            "海淀南路", "丹棱街", "科学院南路", "知春路", "学院路",
            # 扩展区域（万柳/西苑/上地）
            "西土城路", "蓟门桥路", "万柳中路", "蓝靛厂路", "长春桥路",
            "远大路", "巴沟路", "西苑路", "农大南路", "马连洼路",
            "上地南路", "信息路", "软件园路", "西二旗大街", "清河中街",
            "朱房路", "学知桥路", "蓟门里南路", "北太平庄路", "新街口外大街",
            # 西四环/紫竹院片区
            "德胜门外大街", "西直门外大街", "三里河路", "车公庄西路", "紫竹院路",
            "花园桥路", "阜成路", "定慧寺路", "西四环北路", "杏石口路",
            # 世纪城/蓝靛厂片区
            "远大西路", "世纪城路", "苏州桥路", "万泉河桥路", "稻香园路",
            "万柳东路", "万柳西路", "蓝靛厂南路", "火器营路", "板井路"
        ]
        road_name_cache = base_roads.copy()
        # 补全至50条
        #while len(road_name_cache) < ROAD_COUNT_TARGET:
        #    road_name_cache.append(f"海淀{len(road_name_cache)+1}路")
        # 保存缓存
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(road_list_path, "w", encoding="utf-8") as f:
            json.dump(road_name_cache, f, ensure_ascii=False, indent=2)
    
    # 生成roadname.txt
    if not os.path.exists(road_name_txt_path):
        save_road_name_to_txt()
    print(f"初始化道路列表完成，共{len(road_name_cache)}条道路")

def save_road_name_to_txt():
    """保存道路名到txt文件"""
    global road_name_cache
    try:
        road_name_txt_path = os.path.join(SAVE_DIR, "roadname.txt")
        with open(road_name_txt_path, "w", encoding="utf-8") as f:
            for i, road_name in enumerate(road_name_cache):
                f.write(f"Road{i+1}: {road_name}\n")
        print(f"道路名称已保存至：{road_name_txt_path}")
    except Exception as e:
        print(f"保存roadname.txt失败：{e}")

def get_mock_traffic_data():
    """生成模拟交通数据（应急用）"""
    global road_name_cache
    traffic_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "congestion_ratios": {}
    }
    # 模拟高峰时段拥堵（7-9点，17-19点拥堵比例高）
    hour = datetime.now().hour
    base_ratio = 0.0
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        base_ratio = random.uniform(20, 60)  # 高峰拥堵20-60%
    else:
        base_ratio = random.uniform(0, 20)   # 平峰拥堵0-20%
    # 为每条道路生成随机拥堵比例
    for road_name in road_name_cache:
        traffic_data["congestion_ratios"][road_name] = base_ratio + random.uniform(-5, 5)
    return traffic_data

def get_real_traffic_data():
    """获取真实交通数据（API调用）"""
    global road_name_cache
    traffic_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "congestion_ratios": {name: 0.0 for name in road_name_cache}
    }
    # 尝试城市级API（更稳定）
    try:
        url = "https://restapi.amap.com/v3/traffic/status/city"
        params = {
            "key": GAODE_AK,
            "city": CITY_CODE,
            "output": "json"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 城市级API返回：{data}")
        
        if data.get("status") == "1":
            evaluation = data.get("trafficinfo", {}).get("evaluation", {})
            try:
                congested = float(evaluation.get("congested", "0.00%").replace("%", ""))
                blocked = float(evaluation.get("blocked", "0.00%").replace("%", ""))
                total_congestion = congested + blocked
                # 随机分配到各道路（模拟差异）
                for name in traffic_data["congestion_ratios"]:
                    traffic_data["congestion_ratios"][name] = total_congestion + random.uniform(-3, 3)
            except:
                pass
            return traffic_data
        else:
            print(f"城市级API错误：{data.get('info')}（{data.get('infocode')}）")
    except Exception as e:
        print(f"城市级API请求失败：{e}")
    
    # 尝试矩形区域API
    try:
        url = "https://restapi.amap.com/v3/traffic/status/rectangle"
        params = {
            "key": GAODE_AK,
            "rectangle": RECTANGLE_RANGE,
            "output": "json"
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"矩形区域API返回：{data}")
        
        if data.get("status") == "1":
            evaluation = data.get("trafficinfo", {}).get("evaluation", {})
            try:
                congested = float(evaluation.get("congested", "0.00%").replace("%", ""))
                blocked = float(evaluation.get("blocked", "0.00%").replace("%", ""))
                total_congestion = congested + blocked
                for name in traffic_data["congestion_ratios"]:
                    traffic_data["congestion_ratios"][name] = total_congestion + random.uniform(-3, 3)
            except:
                pass
            return traffic_data
        else:
            print(f"矩形区域API错误：{data.get('info')}（{data.get('infocode')}）")
    except Exception as e:
        print(f"矩形区域API请求失败：{e}")
    
    return None

def get_traffic_data():
    """统一获取交通数据（优先真实，失败则模拟）"""
    if not USE_MOCK_DATA:
        real_data = get_real_traffic_data()
        if real_data is not None:
            return real_data
        print("真实数据获取失败，启用模拟数据...")
    return get_mock_traffic_data()

def save_pems_format(data):
    """保存为PEMS格式的CSV文件（增强权限处理）"""
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        csv_filename = os.path.join(SAVE_DIR, f"bj_pems_traffic_{datetime.now().strftime('%Y%m%d')}.csv")
        is_first_write = not os.path.exists(csv_filename)
        
        # 构建行数据（限制数值范围0-100）
        row_data = [data["timestamp"]]
        for road_name in road_name_cache:
            ratio = data["congestion_ratios"].get(road_name, 0.0)
            ratio = max(0.0, min(100.0, ratio))  # 限制0-100%
            row_data.append(round(ratio, 2))  # 保留两位小数
        
        # 写入CSV（使用try-except重试一次）
        for _ in range(2):
            try:
                with open(csv_filename, "a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    if is_first_write:
                        header = ["timestamp"] + road_name_cache
                        writer.writerow(header)
                    writer.writerow(row_data)
                print(f"数据已保存至：{csv_filename}")
                return True
            except PermissionError:
                time.sleep(1)  # 等待1秒重试
                continue
        
        print(f"保存CSV失败：权限被拒绝（重试后仍失败）")
        return False
    except Exception as e:
        print(f"保存CSV失败：{e}")
        return False

if __name__ == "__main__":
    print("===== 初始化道路列表 =====")
    init_road_cache()
    
    print(f"\n===== 开始采集PEMS格式交通数据 =====")
    print(f"采集模式：{'模拟数据' if USE_MOCK_DATA else '真实数据（失败则模拟）'}")
    print(f"采集道路数量：{ROAD_COUNT_TARGET}条")
    print(f"采集间隔：{INTERVAL//60}分钟")
    print(f"单日最大采集次数：{MAX_COLLECT_COUNT}次")
    print(f"数据保存目录：{os.path.abspath(SAVE_DIR)}")
    print("按Ctrl+C停止采集\n")
    
    collect_count = 0
    try:
        while collect_count < MAX_COLLECT_COUNT:
            start_time = time.time()
            # 获取交通数据
            traffic_data = get_traffic_data()
            # 保存数据
            if traffic_data and save_pems_format(traffic_data):
                collect_count += 1
                print(f"已采集：{collect_count}/{MAX_COLLECT_COUNT}次")
            else:
                print("获取/保存数据失败，不计入次数")
            
            # 计算等待时间（保证5分钟间隔）
            elapsed_time = time.time() - start_time
            wait_time = max(0, INTERVAL - elapsed_time)
            if collect_count < MAX_COLLECT_COUNT and wait_time > 0:
                mins, secs = divmod(wait_time, 60)
                print(f"\n等待{int(mins)}分{int(secs)}秒后进行下一次采集...")
                time.sleep(wait_time)
        
        print(f"\n===== 单日采集完成（{MAX_COLLECT_COUNT}次） =====")
    except KeyboardInterrupt:
        print(f"\n===== 手动停止采集（已采集{collect_count}次） =====")
    except Exception as e:
        print(f"\n===== 采集异常：{e}（已采集{collect_count}次） =====")
    finally:
        save_road_name_to_txt()
        print("\n程序结束，道路名文件已保存。")