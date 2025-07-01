from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # You can add specific origins if needed

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def visible(p, threshold=0.3):
    return p.visibility > threshold

def classify_posture(landmarks):
    if landmarks is None:
        return "no person detected", 0

    try:
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    except:
        return "no person detected", 0

    visible_parts = {
        'nose': visible(nose),
        'l_shoulder': visible(l_shoulder),
        'r_shoulder': visible(r_shoulder),
        'l_hip': visible(l_hip),
        'r_hip': visible(r_hip),
        'l_ankle': visible(l_ankle),
        'r_ankle': visible(r_ankle),
    }

    vis_count = sum(visible_parts.values())

    if vis_count < 5:
        return "no person detected", vis_count

    orientation = "unknown"
    ankles_y = [p.y for k, p in zip(['l_ankle', 'r_ankle'], [l_ankle, r_ankle]) if visible_parts[k]]
    hips_y = [p.y for k, p in zip(['l_hip', 'r_hip'], [l_hip, r_hip]) if visible_parts[k]]
    shoulders_y = [p.y for k, p in zip(['l_shoulder', 'r_shoulder'], [l_shoulder, r_shoulder]) if visible_parts[k]]

    if visible_parts['nose'] and ankles_y:
        orientation = "foot_first" if nose.y < min(ankles_y) else "head_first"
    elif hips_y and ankles_y:
        orientation = "head_first" if sum(hips_y)/len(hips_y) < min(ankles_y) else "foot_first"
    elif visible_parts['nose'] and hips_y:
        orientation = "foot_first" if nose.y < sum(hips_y)/len(hips_y) else "head_first"
    elif visible_parts['nose'] and len(shoulders_y) == 2:
        shoulder_y = sum(shoulders_y) / 2
        orientation = "foot_first" if nose.y < shoulder_y else "head_first"
    elif hips_y:
        orientation = "head_first"
    elif ankles_y:
        orientation = "foot_first"

    facing = "unknown"
    if visible_parts['l_shoulder'] and visible_parts['r_shoulder']:
        x_diff = abs(l_shoulder.x - r_shoulder.x)
        if x_diff < 0.05:
            facing = "decubitus_left" if l_shoulder.x < r_shoulder.x else "decubitus_right"
        elif visible_parts['nose']:
            mid_x = (l_shoulder.x + r_shoulder.x) / 2
            facing = "prone" if nose.x < mid_x else "supine"
    elif visible_parts['l_hip'] and visible_parts['r_hip']:
        x_diff = abs(l_hip.x - r_hip.x)
        if x_diff < 0.05:
            facing = "decubitus_left" if l_hip.x < r_hip.x else "decubitus_right"

    posture = "unknown"
    if orientation != "unknown" and facing != "unknown":
        posture = f"{orientation}_{facing}"
    elif orientation != "unknown":
        posture = orientation

    return posture, vis_count

# Tracking variables
prev_posture = None
stable_posture = None
freeze_count = 0
FREEZE_THRESHOLD = 10

def draw_landmarks(image, landmarks):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_drawing.draw_landmarks(
        image_rgb,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

@app.route('/detect-posture', methods=['POST'])
def detect_posture():
    global prev_posture, stable_posture, freeze_count
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        with mp_pose.Pose(static_image_mode=True, model_complexity=1,
                         min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
            results = pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                posture, vis_count = classify_posture(landmarks)

                if posture not in ["no person detected", "unknown"]:
                    if posture == prev_posture:
                        freeze_count += 1
                        if freeze_count >= FREEZE_THRESHOLD:
                            stable_posture = posture
                    else:
                        freeze_count = 0
                        stable_posture = None
                    prev_posture = posture
                else:
                    freeze_count = 0
                    prev_posture = None
                    stable_posture = None

                annotated_image = draw_landmarks(image_bgr.copy(), results.pose_landmarks)
                cv2.putText(annotated_image, f"Posture: {posture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if stable_posture:
                    cv2.putText(annotated_image, f"Stable: {stable_posture}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                if vis_count < 5 and posture != "no person detected":
                    cv2.putText(annotated_image, "Low Visibility!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                _, buffer = cv2.imencode('.jpg', annotated_image)
                annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')

                return jsonify({
                    'posture': stable_posture if stable_posture else posture,
                    'current_posture': posture,
                    'stable_posture': stable_posture,
                    'visibility_count': vis_count,
                    'freeze_count': freeze_count,
                    'annotated_image': f'data:image/jpeg;base64,{annotated_image_base64}',
                    'success': True
                })
            else:
                freeze_count = 0
                prev_posture = None
                stable_posture = None

                return jsonify({
                    'posture': 'no person detected',
                    'current_posture': 'no person detected',
                    'stable_posture': None,
                    'visibility_count': 0,
                    'freeze_count': 0,
                    'annotated_image': None,
                    'success': True
                }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5002, host='0.0.0.0')
