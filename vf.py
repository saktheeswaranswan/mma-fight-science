
from ultralytics import YOLO
import cv2, json
import numpy as np
import os

# Load pretrained YOLOv11 pose model
model = YOLO("yolo11x-pose.pt")  # official large pose model:contentReference[oaicite:3]{index=3}

cap = cv2.VideoCapture("vaddii.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 1280, 720
out = cv2.VideoWriter("output/annoddddjtated_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))
pose_data = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistent input size
    frame = cv2.resize(frame, (width, height))

    # Run pose estimation with tracking to keep IDs across frames:contentReference[oaicite:4]{index=4}
    results = model.track(frame, task="pose", persist=True)
    res = results[0]  # current frame result
    annotated_frame = res.plot()  # draw skeleton

    # Compute timestamp (in seconds) for this frame
    timestamp = frame_count / fps

    # If at least one person detected, extract data
    if res.boxes is not None and len(res.boxes.xyxy):
        # Get tracking IDs (or None if disabled)
        track_ids = res.boxes.id.int().cpu().numpy().tolist() \
                    if res.boxes.id is not None else [None]*len(res.boxes.xyxy)

        # Iterate detected persons
        for i, bbox in enumerate(res.boxes.xyxy):
            person_id = int(track_ids[i]) if track_ids[i] is not None else None
            # Extract 17 keypoints [x, y] for person i
            keypoints = res.keypoints.xy[i].cpu().numpy().tolist()
            # Convert bbox tensor to list
            bbox_xyxy = bbox.cpu().numpy().tolist()

            # -- Compute limb vectors and magnitudes for arms and legs --
            # (Define COCO indices: LShoulder=5, LElbow=7, LWrist=9; RShoulder=6, RElbow=8, RWrist=10;
            #                       LHip=11, LKnee=13, LAnkle=15;  RHip=12, RKnee=14, RAnkle=16)
            vec_data = {}
            pts = np.array(keypoints)  # 17x2 array of (x,y)
            # Left arm: shoulder->elbow and elbow->wrist
            L_shldr, L_elbow, L_wrist = pts[5], pts[7], pts[9]
            v1 = L_elbow - L_shldr;   v2 = L_wrist - L_elbow
            # Right arm
            R_shldr, R_elbow, R_wrist = pts[6], pts[8], pts[10]
            v3 = R_elbow - R_shldr;   v4 = R_wrist - R_elbow
            # Left leg
            L_hip, L_knee, L_ankle = pts[11], pts[13], pts[15]
            v5 = L_knee - L_hip;      v6 = L_ankle - L_knee
            # Right leg
            R_hip, R_knee, R_ankle = pts[12], pts[14], pts[16]
            v7 = R_knee - R_hip;      v8 = R_ankle - R_knee

            # Store each vector's dx, dy, and magnitude
            vecs = [("L_upper_arm", v1), ("L_lower_arm", v2),
                    ("R_upper_arm", v3), ("R_lower_arm", v4),
                    ("L_thigh", v5), ("L_shin", v6),
                    ("R_thigh", v7), ("R_shin", v8)]
            for name, vec in vecs:
                dx, dy = float(vec[0]), float(vec[1])
                mag = (dx**2 + dy**2)**0.5
                vec_data[name] = {"dx": dx, "dy": dy, "magnitude": mag}

            # Optionally, compute resultant vectors (shoulder->wrist, hip->ankle)
            res_vecs = {
                "L_arm_result": (L_wrist - L_shldr),
                "R_arm_result": (R_wrist - R_shldr),
                "L_leg_result": (L_ankle - L_hip),
                "R_leg_result": (R_ankle - R_hip)
            }
            for name, vec in res_vecs.items():
                dx, dy = float(vec[0]), float(vec[1])
                mag = (dx**2 + dy**2)**0.5
                vec_data[name] = {"dx": dx, "dy": dy, "magnitude": mag}

            # Record frame data including vectors
            frame_data = {
                "frame_id": frame_count,
                "timestamp": timestamp,
                "person_id": person_id,
                "keypoints": keypoints,
                "bounding_box": bbox_xyxy,
                "limb_vectors": vec_data
            }
            pose_data.append(frame_data)

    out.write(annotated_frame)
    frame_count += 1

cap.release()
out.release()
with open("output/pose_data.json", "w") as f:
    json.dump(pose_data, f, indent=4)
print("✅ Annotated video and pose data saved.")

