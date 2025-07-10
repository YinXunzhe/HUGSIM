import json

meta_path = "datasets/zone/3D_data_LSJWK4095NS119733/20250317_161633_1/anno_track_refine_od.json"

# 加载原始 JSON 文件
with open(meta_path, "r", encoding="utf-8") as f:
    data = json.load(f)

od_anno_info = data.get("od_anno_info", {})
if not isinstance(od_anno_info, dict):
    raise ValueError("od_anno_info 字段不是字典类型")

# 提取第 130 帧的信息
partial_data = {}
for i, (frame_name, frame_data) in enumerate(od_anno_info.items()):
    if i == 130:
        break

    # pose = frame_data.get("pose")
    # eventDataTime = frame_data.get("eventDataTime")
    # frame_id = frame_data.get("frame_id")

    # # 只保存完整记录（你也可以选择容错）
    # if pose and eventDataTime and frame_id:
    #     partial_data[frame_name] = {
    #         "pose": pose,
    #         "eventDataTime": eventDataTime,
    #         "frame_id": frame_id
    #     }
    partial_data[frame_name] = frame_data

# 保存到输出文件
with open("datasets/zone/3D_data_LSJWK4095NS119733/20250317_161633_1/anno_track_refine_od_130.json", "w", encoding="utf-8") as f:
    json.dump(partial_data, f, indent=2, ensure_ascii=False)
