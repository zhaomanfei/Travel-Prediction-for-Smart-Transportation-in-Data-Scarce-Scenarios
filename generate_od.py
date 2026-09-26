import os
import xml.etree.ElementTree as ET

# 1. 配置文件路径（请确保和你的文件一致）
xml_file_path = "osm.zgc.xml"  # 你的SUMO路网XML文件
od_file_path = "od.txt"       # 要生成的OD文件（存储有效edge ID）
vehicle_count = 30            # 每对OD的车辆数

# 2. 前置验证：文件存在且非空
def pre_check():
    if not os.path.exists(xml_file_path):
        print(f"❌ 错误：未找到文件 {xml_file_path}")
        print(f"   提示：当前工作目录为 {os.getcwd()}")
        print(f"   请确保 {xml_file_path} 放在该目录下，或填写完整路径（如 D:\\xxx\\osm.zgc.xml）")
        return False
    if os.path.getsize(xml_file_path) == 0:
        print(f"❌ 错误：{xml_file_path} 文件为空，请重新解压.gz压缩包")
        return False
    print(f"✅ 前置验证通过：{xml_file_path} 存在且非空")
    return True

# 3. 解析XML，提取<edge>标签的id属性（有效路段ID）
def extract_edge_ids():
    edge_ids = set()  # 用集合存储edge ID，自动去重，避免重复路段
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ 错误：XML 文件解析失败，原因：{str(e)}")
        print(f"   提示：请确保 {xml_file_path} 是合法的XML格式，而非未解压的.gz文件")
        return None
    
    # 查找<edge>标签（优先递归查找，若失败则遍历根节点所有子节点）
    edge_list = root.findall(".//edge")
    print(f"🔍 递归查找 <edge> 标签，共找到 {len(edge_list)} 个")
    
    # 若递归查找失败，尝试直接遍历根节点子节点（适配特殊XML结构）
    if len(edge_list) == 0:
        print("⚠️  递归查找失败，尝试遍历根节点子节点...")
        for child in root:
            if child.tag == "edge":
                edge_list.append(child)
        print(f"🔍 遍历根节点后，找到 <edge> 标签 {len(edge_list)} 个")
    
    if len(edge_list) == 0:
        print("❌ 错误：未找到任何 <edge> 标签，请检查XML文件格式")
        return None
    
    # 提取<edge>的id属性，打印部分结果排查
    for idx, edge in enumerate(edge_list):
        edge_id = edge.get("id")  # 核心修改：提取<edge>自身的id属性（有效路段ID）
        
        # 打印前20个标签的提取结果，方便排查
        if idx < 20:
            print(f"   第 {idx+1} 个 <edge>：id={edge_id}")
        
        # 仅添加非空、非空字符串的edge ID
        if edge_id and edge_id.strip():
            edge_ids.add(edge_id.strip())
    
    print(f"✅ 路段ID提取完成，共获取去重后有效路段 {len(edge_ids)} 个")
    return list(edge_ids)

# 4. 生成od.txt文件（存储edge ID，格式适配SUMO的from/to配置）
def generate_od_file(edge_list):
    if not edge_list or len(edge_list) < 2:
        print("❌ 错误：有效路段ID数量不足（少于2个），无法生成OD记录")
        return False
    
    try:
        with open(od_file_path, "w", encoding="utf-8") as f:
            # 写入注释（明确标注为路段ID，避免混淆节点ID）
            f.write("# 自动生成的OD矩阵文件（存储SUMO有效路段edge ID）\n")
            f.write("# 格式：起点路段ID  终点路段ID  车辆数量\n")
            f.write(f"# 生成时间：{os.path.getctime(__file__)}（仅作标记）\n\n")
            
            # 生成OD记录（两两配对，避免自环（起点=终点））
            record_count = 0
            for start_id in edge_list:
                for end_id in edge_list:
                    if start_id != end_id:  # 避免起点和终点为同一条路段
                        # 用两个制表符分隔，格式更规范，SUMO工具兼容
                        f.write(f"{start_id}\t{end_id}\t{vehicle_count}\n")
                        record_count += 1
        
        print(f"✅ OD文件生成成功！")
        print(f"   文件路径：{os.path.abspath(od_file_path)}")
        print(f"   生成OD记录：{record_count} 条")
        print(f"   文件大小：{os.path.getsize(od_file_path)} 字节")
        return True
    except Exception as e:
        print(f"❌ 错误：OD文件写入失败，原因：{str(e)}")
        return False

# 5. 主执行流程
if __name__ == "__main__":
    # 步骤1：前置验证
    if not pre_check():
        exit(1)
    
    # 步骤2：提取edge ID（核心修改：替换节点ID为路段ID）
    edge_list = extract_edge_ids()
    if not edge_list:
        exit(1)
    
    # 步骤3：生成OD文件
    if not generate_od_file(edge_list):
        exit(1)