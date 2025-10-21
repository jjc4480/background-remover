"""
MediaPipe 기반 얼굴 및 포즈 감지 모듈
정확한 얼굴 랜드마크와 신체 포즈를 감지하여 더 정확한 크롭 제공
"""
import os
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import logging

# Lambda 환경에서 MediaPipe 모델을 /tmp에 다운로드하도록 설정
if os.environ.get('AWS_LAMBDA_FUNCTION_NAME'):
    os.environ['MEDIAPIPE_RESOURCE_DIR'] = '/tmp/mediapipe'
    os.makedirs('/tmp/mediapipe', exist_ok=True)

logger = logging.getLogger(__name__)


class MediaPipeFaceDetector:
    """MediaPipe를 사용한 얼굴 및 포즈 감지 클래스"""

    def __init__(self):
        """초기화"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.face_mesh = None
        self.pose = None

    def _ensure_models(self):
        """모델 초기화 (lazy loading)"""
        if self.face_mesh is None:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        if self.pose is None:
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )

    def detect_face_and_body(self, image, filename="unknown"):
        """
        얼굴과 신체 포즈 감지

        Args:
            image: PIL Image (RGBA)
            filename: 로깅용 파일명

        Returns:
            dict: {
                'face_bbox': (x, y, w, h),  # 얼굴 바운딩 박스
                'face_center': (cx, cy),     # 얼굴 중심
                'left_ear': (x, y),          # 왼쪽 귀 위치
                'right_ear': (x, y),         # 오른쪽 귀 위치
                'nose': (x, y),              # 코 위치
                'chin': (x, y),              # 턱 위치
                'forehead': (x, y),          # 이마 위치
                'hands': [(x, y), ...],      # 손 위치들
                'shoulders': [(x, y), ...],  # 어깨 위치들
            }
        """
        self._ensure_models()

        # PIL to OpenCV
        img_array = np.array(image)
        if img_array.shape[2] == 4:  # RGBA
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img_array

        height, width = img_rgb.shape[:2]
        result = {}

        # 1. 얼굴 랜드마크 감지
        face_results = self.face_mesh.process(img_rgb)

        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark

            # 주요 포인트 추출
            # MediaPipe Face Mesh 인덱스:
            # 10: 이마 중앙 상단
            # 234, 454: 이마 좌우
            # 1: 코 끝
            # 152: 턱 끝
            # 234: 왼쪽 귀 (관자놀이)
            # 454: 오른쪽 귀 (관자놀이)

            nose = landmarks[1]
            chin = landmarks[152]
            forehead = landmarks[10]
            left_ear = landmarks[234]
            right_ear = landmarks[454]

            # 픽셀 좌표로 변환
            result['nose'] = (int(nose.x * width), int(nose.y * height))
            result['chin'] = (int(chin.x * width), int(chin.y * height))
            result['forehead'] = (int(forehead.x * width), int(forehead.y * height))
            result['left_ear'] = (int(left_ear.x * width), int(left_ear.y * height))
            result['right_ear'] = (int(right_ear.x * width), int(right_ear.y * height))

            # 얼굴 바운딩 박스 계산 (모든 랜드마크 기준)
            x_coords = [lm.x * width for lm in landmarks]
            y_coords = [lm.y * height for lm in landmarks]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            result['face_bbox'] = (x_min, y_min, x_max - x_min, y_max - y_min)
            result['face_center'] = ((x_min + x_max) // 2, (y_min + y_max) // 2)

            logger.info(f"[{filename}] ✓ Face detected: bbox={result['face_bbox']}, center={result['face_center']}")
        else:
            logger.warning(f"[{filename}] ✗ No face detected")
            return None

        # 2. 포즈 감지 (손, 어깨)
        pose_results = self.pose.process(img_rgb)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            # 손목 감지 (인덱스 15, 16: 왼쪽/오른쪽 손목)
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]

            hands = []
            # visibility > 0.5 인 경우만 감지된 것으로 간주
            if left_wrist.visibility > 0.5:
                hands.append((int(left_wrist.x * width), int(left_wrist.y * height)))
            if right_wrist.visibility > 0.5:
                hands.append((int(right_wrist.x * width), int(right_wrist.y * height)))

            result['hands'] = hands

            # 어깨 감지 (인덱스 11, 12: 왼쪽/오른쪽 어깨)
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]

            shoulders = []
            if left_shoulder.visibility > 0.5:
                shoulders.append((int(left_shoulder.x * width), int(left_shoulder.y * height)))
            if right_shoulder.visibility > 0.5:
                shoulders.append((int(right_shoulder.x * width), int(right_shoulder.y * height)))

            result['shoulders'] = shoulders

            logger.info(f"[{filename}] ✓ Pose detected: {len(hands)} hands, {len(shoulders)} shoulders")
        else:
            result['hands'] = []
            result['shoulders'] = []
            logger.info(f"[{filename}] ℹ No pose detected")

        return result

    def __del__(self):
        """리소스 정리"""
        if self.face_mesh:
            self.face_mesh.close()
        if self.pose:
            self.pose.close()
