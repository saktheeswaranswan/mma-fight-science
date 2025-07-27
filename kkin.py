import cv2
import json
import numpy as np
import os

# ==== SETTINGS ====
input_video_path = "lve.mp4"
json_path = "pose_data-palparu.json"
output_video_path = "output/bio_pose_output_palparu.mp4"
output_width, output_height = 960, 840
fps = 30
output_duration_sec = 424.8
total_frames = int(fps * output_duration_sec)

# ==== LOAD POSE DATA ====
with open(json_path, "r") as f:
    original_pose_data = json.load(f)

pose_duration_sec = 60 * 93
pose_total_frames = int(pose_duration_sec * fps)
repeat_count = int(np.ceil(total_frames / pose_total_frames))

# Repeat pose data to match video length
wrapped_pose_data = {}
for repeat in range(repeat_count):
    for entry in original_pose_data:
        orig_fid = entry["frame_id"]
        new_fid = orig_fid + repeat * pose_total_frames
        if new_fid < total_frames:
            wrapped_pose_data.setdefault(new_fid, []).append(entry)

# COCO skeleton edges
skeleton_edges = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12)
]

# ==== VIDEO SETUP ====
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise FileNotFoundError("❌ Cannot open video")

os.makedirs("output", exist_ok=True)
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

frame_id = 0
video_frame_id = 0
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while frame_id < total_frames:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video_frame_id = 0
        continue

    # Resize the frame
    orig_h, orig_w = frame.shape[:2]
    scale = output_width / orig_w
    new_h = int(orig_h * scale)
    resized = cv2.resize(frame, (output_width, new_h))

    # Create padded canvas
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    y_offset = (output_height - new_h) // 2
    if 0 <= y_offset < output_height and new_h <= output_height:
        canvas[y_offset:y_offset + new_h, :] = resized
    else:
        resized = cv2.resize(frame, (output_width, output_height))
        canvas = resized.copy()
        scale = output_width / orig_w
        y_offset = 0

    # === DRAW POSE ===
    persons = wrapped_pose_data.get(frame_id, [])

    for person in persons:
        keypoints = np.array(person["keypoints"], dtype=np.float32)
        scaled_keypoints = [(int(x * scale), int(y * scale) + y_offset) for x, y in keypoints]

        # Skip if not enough keypoints are in-frame
        inframe = [pt for pt in scaled_keypoints if 0 <= pt[0] < output_width and 0 <= pt[1] < output_height]
        if len(inframe) < 5:
            continue

        # Draw keypoints
        for pt in scaled_keypoints:
            cv2.circle(canvas, pt, 4, (0, 255, 0), -1)

        # Draw skeleton edges
        for i, j in skeleton_edges:
            try:
                pt1 = scaled_keypoints[i]
                pt2 = scaled_keypoints[j]
                cv2.line(canvas, pt1, pt2, (255, 0, 0), 2)
            except IndexError:
                continue

        # Draw limb vectors (if available)
        vecs = person.get("limb_vectors", {})
        for name, vec in vecs.items():
            dx = vec["dx"] * scale * 0.5
            dy = vec["dy"] * scale * 0.5

            def joint(n): return scaled_keypoints[n]

            try:
                if "L_upper_arm" in name: origin = joint(5)
                elif "L_lower_arm" in name: origin = joint(7)
                elif "R_upper_arm" in name: origin = joint(6)
                elif "R_lower_arm" in name: origin = joint(8)
                elif "L_thigh" in name: origin = joint(11)
                elif "L_shin" in name: origin = joint(13)
                elif "R_thigh" in name: origin = joint(12)
                elif "R_shin" in name: origin = joint(14)
                elif "L_arm_result" in name: origin = joint(5)
                elif "R_arm_result" in name: origin = joint(6)
                elif "L_leg_result" in name: origin = joint(11)
                elif "R_leg_result" in name: origin = joint(12)
                else: continue
            except IndexError:
                continue

            tip = (int(origin[0] + dx), int(origin[1] + dy))
            color = (0, 0, 255) if "result" not in name else (255, 255, 0)
            cv2.arrowedLine(canvas, origin, tip, color, 2, tipLength=0.2)

    # Write frame to output
    out.write(canvas)
    frame_id += 1
    video_frame_id += 1

    if frame_id % 300 == 0:
        print(f"🟢 Processed {frame_id}/{total_frames} frames")

# ==== CLEANUP ====
cap.release()
out.release()
print(f"✅ Video saved to: {output_video_path}")

