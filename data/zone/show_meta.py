import json

meta_path = "/home/sczone/hugsim_workspace/datasets/zone/3D_data_LSJWK4095NS119733/20250304_154844_2/meta.json"

# 加载原始 JSON 文件
with open(meta_path, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = data.get("meta", {})
if not isinstance(meta, dict):
    raise ValueError("meta 字段不是字典类型")

# 提取前 150 帧的信息
partial_data = {}
for i, (frame_name, frame_data) in enumerate(meta.items()):
    if i >= 150:
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
with open("/home/sczone/hugsim_workspace/datasets/zone/3D_data_LSJWK4095NS119733/20250304_154844_2/pose_150_with_time.json", "w", encoding="utf-8") as f:
    json.dump(partial_data, f, indent=2, ensure_ascii=False)
