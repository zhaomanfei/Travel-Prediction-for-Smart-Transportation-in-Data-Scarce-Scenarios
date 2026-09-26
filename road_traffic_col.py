import requests
import csv
import time
import os
from datetime import datetime

# -------------------------- 配置参数 --------------------------
GAODE_AK = "c41e06817e217d9ad8417504938c0e69"  # 替换为你的AK
CITY_CODE = "110000"             # 北京城市编码
RECTANGLE = "116.2900,39.9600;116.3300,39.9900"  # 圆明园西路扩大的经纬度范围
INTERVAL = 600                   # 采集间隔（10分钟）
MAX_COLLECT_COUNT = 90           # 单日最大采集次数
SAVE_FILENAME = os.path.join(os.path.expanduser("~"), "Desktop", "traffic_data.csv")  # 保存到桌面，便于查找
# 历史路况记录
history_traffic = {"status": None, "expedite": None}
# -------------------------------------------------------------

def get_gaode_traffic():
    """调用高德交通态势查询（兼容城市级和区域级数据）"""
    # 先尝试区域级查询
    url_rect = "https://restapi.amap.com/v3/traffic/status/rectangle"
    params_rect = {
        "key": GAODE_AK,
        "rectangle": RECTANGLE,
        "output": "json"
    }
    try:
        resp = requests.get(url_rect, params_rect, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API返回数据：{data}")
        
        if data.get("status") == "1":
            trafficinfo = data.get("trafficinfo", {})
            # 解析区域级roads数据
            roads = trafficinfo.get("roads", [])
            if roads:
                road = roads[0]  # 取第一条道路
                return {
                    "采集时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "道路名称": road.get("name", "未知道路"),
                    "整体路况描述": road.get("status_desc", "未知"),
                    "畅通比例(%)": trafficinfo.get("evaluation", {}).get("expedite", "未知"),
                    "拥堵比例(%)": trafficinfo.get("evaluation", {}).get("congested", "未知"),
                    "路况等级": road.get("status", trafficinfo.get("evaluation", {}).get("status", "未知")),
                    "路况详情": f"车速：{road.get('speed', 0)}km/h，拥堵长度：{road.get('congestion_length', 0)}m"
                }
            # 解析城市级整体数据（roads为空时）
            else:
                return {
                    "采集时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "道路名称": "北京海淀区（圆明园西路附近）",
                    "整体路况描述": trafficinfo.get("description", "未知"),
                    "畅通比例(%)": trafficinfo.get("evaluation", {}).get("expedite", "未知"),
                    "拥堵比例(%)": trafficinfo.get("evaluation", {}).get("congested", "未知"),
                    "路况等级": trafficinfo.get("evaluation", {}).get("status", "未知"),
                    "路况详情": trafficinfo.get("evaluation", {}).get("description", "未知")
                }
        else:
            # 区域查询失败，尝试城市级查询
            url_city = "https://restapi.amap.com/v3/traffic/status/city"
            params_city = {"key": GAODE_AK, "city": CITY_CODE, "output": "json"}
            resp_city = requests.get(url_city, params_city, timeout=10)
            data_city = resp_city.json()
            if data_city.get("status") == "1":
                trafficinfo = data_city.get("trafficinfo", {})
                return {
                    "采集时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "道路名称": "北京市",
                    "整体路况描述": trafficinfo.get("description", "未知"),
                    "畅通比例(%)": trafficinfo.get("evaluation", {}).get("expedite", "未知"),
                    "拥堵比例(%)": trafficinfo.get("evaluation", {}).get("congested", "未知"),
                    "路况等级": trafficinfo.get("evaluation", {}).get("status", "未知"),
                    "路况详情": trafficinfo.get("evaluation", {}).get("description", "未知")
                }
            else:
                print(f"高德API错误：{data.get('info', '未知错误')}（错误码：{data.get('infocode')}）")
                return None
    except Exception as e:
        print(f"请求失败：{str(e)}")
        return None

def save_to_csv(data):
    """保存数据到CSV（桌面路径）"""
    if not data:
        return False
    fieldnames = ["采集时间", "道路名称", "整体路况描述", "畅通比例(%)", "拥堵比例(%)", "路况等级", "路况详情"]
    try:
        # 检查文件是否存在，不存在则写表头
        file_exists = os.path.exists(SAVE_FILENAME)
        with open(SAVE_FILENAME, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        print(f"数据已保存至：{SAVE_FILENAME}")
        return True
    except Exception as e:
        print(f"保存失败：{str(e)}")
        return False

def check_traffic_change(current_data):
    """检查路况变化"""
    global history_traffic
    if not current_data:
        return
    current_status = current_data.get("路况等级")
    current_expedite = current_data.get("畅通比例(%)")
    change_flag = False
    change_info = ""
    
    # 对比路况等级
    if history_traffic["status"] and history_traffic["status"] != current_status:
        change_flag = True
        change_info += f"路况等级：{history_traffic['status']} → {current_status}；"
    
    # 对比畅通比例（变化>5%）
    if history_traffic["expedite"] and current_expedite != "未知":
        try:
            h_pct = float(history_traffic["expedite"].replace("%", "")) if "%" in history_traffic["expedite"] else 0
            c_pct = float(current_expedite.replace("%", "")) if "%" in current_expedite else 0
            if abs(h_pct - c_pct) > 5:
                change_flag = True
                change_info += f"畅通比例：{history_traffic['expedite']} → {current_expedite}；"
        except:
            pass
    
    if change_flag:
        print(f"\n===== 【路况变化】 =====\n{change_info}\n========================")
    else:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 路况暂未明显变化")
    
    history_traffic["status"] = current_status
    history_traffic["expedite"] = current_expedite

if __name__ == "__main__":
    print(f"===== 开始采集北京海淀区路况数据 =====")
    print(f"采集间隔：{INTERVAL//60}分钟 | 保存路径：{SAVE_FILENAME}")
    print("按Ctrl+C停止采集\n")
    
    collect_count = 0
    try:
        while collect_count < MAX_COLLECT_COUNT:
            traffic_data = get_gaode_traffic()
            if traffic_data:
                if save_to_csv(traffic_data):
                    collect_count += 1
                    print(f"已采集：{collect_count}/{MAX_COLLECT_COUNT}次")
                    check_traffic_change(traffic_data)
                else:
                    print("数据保存失败，不计入次数")
            else:
                print("获取数据失败，不计入次数")
            
            if collect_count < MAX_COLLECT_COUNT:
                print(f"\n等待{INTERVAL//60}分钟后下一次采集...")
                time.sleep(INTERVAL)
        
        print(f"\n===== 单日采集完成（{MAX_COLLECT_COUNT}次） =====")
    except KeyboardInterrupt:
        print(f"\n===== 手动停止采集（已采集{collect_count}次） =====")
    except Exception as e:
        print(f"\n===== 采集异常：{str(e)}（已采集{collect_count}次） =====")