"""
배경 제거 핵심 로직 모듈
Lambda와 로컬 환경에서 공통으로 사용
"""
import os

# 리소스 사용량 제한 (Lambda 환경에서도 적용 - 메모리 최적화)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove, new_session
import cv2
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)

# MediaPipe 감지기 (lazy import)
_mediapipe_detector = None

def get_mediapipe_detector():
    """MediaPipe 감지기 가져오기 (lazy loading)"""
    global _mediapipe_detector

    if _mediapipe_detector is None:
        try:
            # 절대 import 시도
            try:
                from mediapipe_face_detector import MediaPipeFaceDetector
            except ImportError:
                # 상대 import 시도
                from .mediapipe_face_detector import MediaPipeFaceDetector

            _mediapipe_detector = MediaPipeFaceDetector()
            logger.info("MediaPipe detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MediaPipe: {e}")
            _mediapipe_detector = False  # 실패 표시
    return _mediapipe_detector if _mediapipe_detector is not False else None

# 이미지 처리 설정
MAX_DIMENSION = 1500          # 입력 이미지 최대 가로/세로

# 상반신 표준화 설정 (증명사진 스타일)
STANDARD_OUTPUT_WIDTH = 600   # 표준 출력 너비
STANDARD_OUTPUT_HEIGHT = 800  # 표준 출력 높이
# 최종 버전별 고정 크기
VERSION1_TARGET_WIDTH = 600   # 버전1: 상반신 고정 너비 (증명사진) - 480 -> 600
VERSION1_TARGET_HEIGHT = 640  # 버전1: 상반신 고정 높이
VERSION2_TARGET_WIDTH = 300   # 버전2: 원형 고정 너비 (시상 페이지)
VERSION2_TARGET_HEIGHT = 500  # 버전2: 원형 고정 높이 (머리카락 포함) - 400 -> 500


class BackgroundRemover:
    """배경 제거 처리 클래스"""

    def __init__(self, model='birefnet-general'):
        """
        초기화

        Args:
            model: 사용할 모델명 (birefnet-general, u2net 등)
        """
        self.model = model
        self.session = None
        self._face_cascade = None

    def _create_session(self):
        """모델 세션 생성"""
        if self.session is None:
            try:
                logger.info(f"Creating {self.model} session...")
                self.session = new_session(self.model)
                logger.info(f"Session created successfully: {type(self.session)}")
            except Exception as e:
                logger.warning(f"Failed to create {self.model} session: {e}")
                logger.info("Using default model...")
                self.session = None

    def remove_background(self, input_image, filename="unknown"):
        """
        이미지 배경 제거 (최종 2가지 버전 생성)

        Args:
            input_image: PIL Image 객체
            filename: 파일명 (로깅용)

        Returns:
            tuple: (version1, version2)
                - version1: VS 화면용 어깨선 상반신 (MediaPipe 기반, 일관된 크기)
                - version2: 시상 내역용 얼굴 원형 크롭 (MediaPipe 기반, 손 포즈 제외)
        """
        # 세션 생성 (최초 1회)
        self._create_session()

        # 메모리 최적화: 큰 이미지를 적절한 크기로 줄이기
        original_width, original_height = input_image.size

        if original_width > MAX_DIMENSION or original_height > MAX_DIMENSION:
            # 비율 유지하면서 리사이즈
            ratio = min(MAX_DIMENSION / original_width, MAX_DIMENSION / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            input_image = input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"[{filename}] Resized for memory: {original_width}x{original_height} -> {new_width}x{new_height}")

        # 배경 제거
        logger.info(f"[{filename}] Removing background...")
        if self.session:
            output_image = remove(input_image, session=self.session)
        else:
            output_image = remove(input_image)

        # 후처리 적용
        output_image = self._apply_post_processing(output_image)

        # 여백 최소화
        output_image = self._crop_to_content(output_image)

        # MediaPipe 감지기 사용
        detector = get_mediapipe_detector()

        if detector is None:
            logger.warning(f"[{filename}] MediaPipe not available, using Haar Cascade fallback")
            return self._haar_cascade_fallback(output_image, filename)

        # MediaPipe 감지
        detection = detector.detect_face_and_body(output_image, filename=filename)

        if detection is None:
            logger.warning(f"[{filename}] MediaPipe detection failed, using Haar Cascade fallback")
            return self._haar_cascade_fallback(output_image, filename)

        # MediaPipe 기반 최종 버전 생성
        version1 = self._create_final_upper_body(output_image.copy(), detection, filename=filename)
        version2 = self._create_final_circular_crop_wide(version1.copy(), detection, filename=filename)

        return version1, version2

    def _haar_cascade_fallback(self, image, filename="unknown"):
        """Haar Cascade 기반 fallback 처리 (MediaPipe 사용 불가 시)"""
        version1 = self._normalize_upper_body(image, filename=filename)
        version2 = self._create_circular_face_crop_from_normalized(version1.copy(), filename=filename)
        version1 = self._resize_to_fixed_size(version1, VERSION1_TARGET_WIDTH, VERSION1_TARGET_HEIGHT)
        version2 = self._resize_to_fixed_size(version2, VERSION2_TARGET_WIDTH, VERSION2_TARGET_HEIGHT)
        return version1, version2

    def _crop_to_content(self, image):
        """
        투명한 여백을 최소화하여 크롭

        Args:
            image: PIL Image (RGBA)

        Returns:
            PIL Image: 크롭된 이미지
        """
        logger.info("Cropping to content...")

        # numpy 배열로 변환
        img_array = np.array(image)

        if len(img_array.shape) < 3 or img_array.shape[2] != 4:
            logger.warning("Image is not RGBA format, skipping crop")
            return image

        # 알파 채널에서 불투명 픽셀 찾기
        alpha = img_array[:, :, 3]
        non_zero = np.where(alpha > 0)

        if len(non_zero[0]) == 0 or len(non_zero[1]) == 0:
            logger.warning("No non-transparent pixels found")
            return image

        # 바운딩 박스 계산
        top = non_zero[0].min()
        bottom = non_zero[0].max()
        left = non_zero[1].min()
        right = non_zero[1].max()

        # 약간의 패딩 추가 (선택적, 너무 꽉 차지 않도록)
        padding = 5
        height, width = img_array.shape[:2]
        top = max(0, top - padding)
        bottom = min(height, bottom + padding)
        left = max(0, left - padding)
        right = min(width, right + padding)

        # 크롭
        cropped = image.crop((left, top, right, bottom))
        logger.info(f"Cropped from {image.size} to {cropped.size}")

        return cropped

    def _resize_to_fixed_size(self, image, target_width, target_height):
        """
        이미지를 고정 크기로 리사이즈 (비율 유지, 머리카락 안 잘림)

        Args:
            image: PIL Image (RGBA)
            target_width: 목표 너비 (픽셀)
            target_height: 목표 높이 (픽셀)

        Returns:
            PIL Image: 리사이즈된 이미지
        """
        # 비율을 유지하면서 목표 크기 안에 들어가도록 리사이즈
        width_ratio = target_width / image.width
        height_ratio = target_height / image.height
        scale = min(width_ratio, height_ratio)  # 작은 쪽 기준 (머리카락 안 잘림)

        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 투명 배경 캔버스에 중앙 배치 (얼굴이 센터에)
        canvas = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        canvas.paste(resized, (left, top), resized)

        logger.info(f"Resized to fixed size: {canvas.size} (no crop, content preserved)")
        return canvas

    def _apply_post_processing(self, image):
        """
        엣지 개선을 위한 후처리

        Args:
            image: PIL Image (RGBA)

        Returns:
            PIL Image: 후처리가 적용된 이미지
        """
        logger.info("Applying enhanced post-processing...")

        # numpy 배열로 변환
        img_array = np.array(image)

        if len(img_array.shape) < 3 or img_array.shape[2] != 4:
            logger.warning("Image is not RGBA format, skipping post-processing")
            return image

        # 알파 채널 추출
        alpha = img_array[:, :, 3].astype(np.float32) / 255.0

        # Trimap 생성 (확실한 전경/배경/불확실 영역 구분)
        trimap = np.zeros_like(alpha, dtype=np.uint8)
        trimap[alpha > 0.8] = 255  # 확실한 전경
        trimap[alpha < 0.1] = 0    # 확실한 배경
        trimap[(alpha >= 0.1) & (alpha <= 0.8)] = 128  # 불확실 영역

        # 가우시안 필터로 알파 채널 부드럽게
        alpha_smooth = gaussian_filter(alpha, sigma=1.5)

        # 엣지 보존 스무딩
        alpha_uint8 = (alpha_smooth * 255).astype(np.uint8)

        # Bilateral 필터로 엣지 보존하며 노이즈 제거
        alpha_bilateral = cv2.bilateralFilter(alpha_uint8, 9, 75, 75)

        # float로 다시 변환
        alpha_final = alpha_bilateral.astype(np.float32) / 255.0

        # 엣지 페더링 (자연스러운 전환)
        mask_binary = (alpha_final > 0.5).astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)

        if dist_transform.max() > 0:
            feather_radius = 3
            feather_mask = np.minimum(dist_transform / feather_radius, 1.0)
            alpha_final = alpha_final * feather_mask

        # 알파 채널 대비 향상 (더 깨끗한 엣지)
        alpha_final = np.clip(alpha_final * 1.2 - 0.1, 0, 1)

        # 알파 채널 업데이트
        img_array[:, :, 3] = (alpha_final * 255).astype(np.uint8)

        # PIL Image로 변환
        output_image = Image.fromarray(img_array, 'RGBA')

        # 색상 향상 (약간의 채도 증가)
        enhancer = ImageEnhance.Color(output_image)
        output_image = enhancer.enhance(1.05)

        return output_image

    def _detect_face_and_body(self, image, filename="unknown"):
        """
        얼굴과 상체 영역 감지 (OpenCV 기반)

        Args:
            image: PIL Image (RGBA)
            filename: 파일명 (로깅용)

        Returns:
            dict: {'face': (x, y, w, h), 'shoulder_y': int} 또는 None
        """
        # PIL to OpenCV
        img_array = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출 (cascade를 캐싱하여 매번 XML 파싱 방지)
        if self._face_cascade is None:
            self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            logger.warning(f"[{filename}] No face detected")
            return None

        # 가장 상단에 있고 큰 얼굴 선택 (상단 우선, 크기 차선)
        face = min(faces, key=lambda f: (f[1], -f[2] * f[3]))
        fx, fy, fw, fh = face

        # 어깨선 추정: 얼굴 하단 + 얼굴 높이의 30-40%
        shoulder_y = fy + fh + int(fh * 0.35)

        logger.info(f"[{filename}] Face detected at ({fx}, {fy}, {fw}, {fh}), shoulder_y estimated at {shoulder_y}")

        return {
            'face': face,
            'shoulder_y': shoulder_y
        }

    def _normalize_upper_body(self, image, filename="unknown"):
        """
        상반신을 표준 크기/비율로 정규화 (증명사진 스타일)

        Args:
            image: PIL Image (RGBA)
            filename: 파일명 (로깅용)

        Returns:
            PIL Image: 정규화된 이미지
        """
        logger.info(f"[{filename}] Normalizing upper body to standard size...")

        img_array = np.array(image)

        if len(img_array.shape) < 3 or img_array.shape[2] != 4:
            logger.warning(f"[{filename}] Image is not RGBA format, skipping normalization")
            return image

        # 얼굴 및 어깨 감지
        detection = self._detect_face_and_body(image, filename=filename)

        if detection is None:
            # 인체가 아닌 경우 (제품, 동물 등) - 누끼만 따고 반환
            logger.info(f"[{filename}] ⚠️  No valid face detected - treating as non-human object (product/animal)")
            return image

        fx, fy, fw, fh = detection['face']
        shoulder_y = detection['shoulder_y']

        # 알파 채널에서 사람 영역 찾기
        alpha = img_array[:, :, 3]
        non_zero = np.where(alpha > 10)

        if len(non_zero[0]) == 0:
            return image

        # 상반신 영역 계산 (얼굴 + 어깨선만 살짝)
        head_top = max(0, fy - int(fh * 0.5))  # 얼굴 위 충분한 여백
        upper_body_bottom = min(image.height, shoulder_y + int(fh * 0.3))  # 어깨선 살짝만

        # 좌우 범위: 실제 사람 영역 기준 + 여백
        # 어깨선 근처의 실제 너비 계산
        shoulder_region_start = max(0, shoulder_y - int(fh * 0.2))
        shoulder_region_end = min(image.height, shoulder_y + int(fh * 0.5))
        shoulder_region_alpha = alpha[shoulder_region_start:shoulder_region_end, :]

        # 어깨 영역에서 실제 픽셀 찾기
        shoulder_pixels_rows = np.any(shoulder_region_alpha > 10, axis=0)
        shoulder_indices = np.where(shoulder_pixels_rows)[0]

        if len(shoulder_indices) > 0:
            actual_left = shoulder_indices.min()
            actual_right = shoulder_indices.max()

            # 좌우 여백 추가 (실제 너비의 15%)
            width_margin = int((actual_right - actual_left) * 0.15)
            left = max(0, actual_left - width_margin)
            right = min(image.width, actual_right + width_margin)
        else:
            # fallback: 얼굴 기준
            face_center_x = fx + fw // 2
            crop_width = fw * 2.5
            left = max(0, int(face_center_x - crop_width / 2))
            right = min(image.width, int(face_center_x + crop_width / 2))

        # 상반신만 크롭
        upper_body_image = image.crop((left, head_top, right, upper_body_bottom))

        # 얼굴 크기를 기준으로 스케일 계산 (모든 인물의 얼굴 크기를 동일하게)
        # 목표: 얼굴이 출력 너비의 35-40% 차지
        target_face_width = int(STANDARD_OUTPUT_WIDTH * 0.37)
        scale_factor = target_face_width / max(fw, 1)

        # 리사이즈 (고품질 + 알파 채널 부드럽게)
        new_width = int(upper_body_image.width * scale_factor)
        new_height = int(upper_body_image.height * scale_factor)

        # LANCZOS로 리샘플링 (고품질, 원본 엣지 유지)
        scaled_image = upper_body_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 실제 이미지 높이에 맞게 캔버스 크기 조정 (아래 여백 최소화)
        # 상단 여백 10% + 이미지 높이
        required_height = int(STANDARD_OUTPUT_HEIGHT * 0.10) + new_height
        canvas_height = min(STANDARD_OUTPUT_HEIGHT, max(new_height + 100, required_height))

        # 표준 캔버스 생성
        canvas = Image.new('RGBA', (STANDARD_OUTPUT_WIDTH, canvas_height), (0, 0, 0, 0))

        # 중앙 상단에 배치 (머리가 상단에서 8% 정도 여백)
        paste_x = (STANDARD_OUTPUT_WIDTH - new_width) // 2
        paste_y = int(canvas_height * 0.08)

        # 높이가 캔버스를 넘으면 크롭
        if paste_y + new_height > canvas_height:
            crop_bottom = canvas_height - paste_y
            scaled_image = scaled_image.crop((0, 0, new_width, crop_bottom))

        # 너비가 캔버스를 넘으면 크롭
        if paste_x < 0 or new_width > STANDARD_OUTPUT_WIDTH:
            paste_x = 0
            scaled_image = scaled_image.crop(((new_width - STANDARD_OUTPUT_WIDTH) // 2, 0,
                                              (new_width + STANDARD_OUTPUT_WIDTH) // 2, scaled_image.height))

        # 붙여넣기
        canvas.paste(scaled_image, (paste_x, paste_y), scaled_image)

        # 최종 여백 제거 (투명 영역 크롭)
        canvas = self._crop_to_content(canvas)

        logger.info(f"Normalized from {image.size} to {canvas.size} (scale: {scale_factor:.2f})")

        return canvas

    def _create_circular_face_crop_from_normalized(self, image, filename="unknown"):
        """
        정규화된 이미지(버전1)에서 얼굴 중심 원형 크롭 생성

        Args:
            image: PIL Image (RGBA) - 이미 정규화된 이미지
            filename: 파일명 (로깅용)

        Returns:
            PIL Image: 원형 크롭된 이미지
        """
        logger.info(f"[{filename}] Creating circular face crop from normalized image...")

        img_array = np.array(image)

        if len(img_array.shape) < 3 or img_array.shape[2] != 4:
            logger.warning(f"[{filename}] Image is not RGBA format, returning original")
            return image

        # 얼굴 감지
        detection = self._detect_face_and_body(image, filename=filename)

        if detection is None:
            logger.warning(f"[{filename}] No face detected for circular crop, returning original")
            return image

        fx, fy, fw, fh = detection['face']
        height, width = img_array.shape[:2]

        # 실제 사람 영역의 중심 찾기 (알파 채널 기반)
        alpha = img_array[:, :, 3]
        # 얼굴 높이 범위에서의 실제 픽셀 찾기
        face_region_alpha = alpha[fy:fy+fh, :]
        person_pixels = np.any(face_region_alpha > 10, axis=0)
        person_indices = np.where(person_pixels)[0]

        if len(person_indices) > 0:
            # 실제 사람 영역의 좌우 끝점
            actual_left = person_indices.min()
            actual_right = person_indices.max()
            # 실제 중심
            face_center_x = (actual_left + actual_right) // 2
        else:
            # fallback: 감지된 얼굴 중심
            face_center_x = fx + fw // 2

        face_center_y = fy + fh // 2

        # 원형 반지름: 얼굴 크기에 1.15배 (얼굴 + 약간의 여백)
        circle_radius = int(max(fw, fh) * 1.15)

        # 좌우 크롭 범위: 얼굴과 귀 기준 (머리카락 고려, 포즈 제외)
        # 얼굴 감지된 사각형의 좌우에 여유 추가 (귀 포함)
        # 기본: 얼굴 너비의 0.65배 (귀까지 포함)
        base_crop_width_half = int(fw * 0.65)

        # 머리 영역에서 머리카락 확인 (얼굴 위쪽만)
        head_top = max(0, fy - int(fh * 0.5))  # 머리 위
        head_bottom = min(height, fy)  # 얼굴 시작점까지만
        head_region_alpha = alpha[head_top:head_bottom, :]

        # 머리 영역에서 픽셀이 있는 좌우 범위 찾기
        head_column_has_content = np.any(head_region_alpha > 10, axis=0)
        head_content_indices = np.where(head_column_has_content)[0]

        if len(head_content_indices) > 0:
            # 머리카락 포함한 너비
            head_left = head_content_indices.min()
            head_right = head_content_indices.max()

            # 얼굴 중심에서 머리카락 끝까지의 거리
            left_distance = face_center_x - head_left
            right_distance = head_right - face_center_x

            # 대칭을 위해 큰 쪽 기준으로
            max_distance = max(left_distance, right_distance)

            # 얼굴 너비 기준과 머리카락 기준 중 큰 값 선택 (단, 얼굴 너비의 0.8배 제한)
            crop_width_half = min(max(base_crop_width_half, max_distance), int(fw * 0.8))
            logger.info(f"[{filename}] Hair detected, symmetric crop_width_half: {crop_width_half}")
        else:
            # fallback: 얼굴 너비 기준
            crop_width_half = base_crop_width_half
            logger.info(f"[{filename}] Using base crop_width_half: {crop_width_half}")

        # 얼굴 중심 기준으로 좌우 대칭 크롭
        crop_left = max(0, face_center_x - crop_width_half)
        crop_right = min(width, face_center_x + crop_width_half)

        # 원형 + 좌우 직선 마스크 생성
        height, width = img_array.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # 1. 전체 원형 그리기
        cv2.circle(mask, (face_center_x, face_center_y), circle_radius, 255, -1)

        # 2. 좌우를 직사각형으로 제거 (귀 밖 영역)
        # 왼쪽 제거
        mask[:, :crop_left] = 0
        # 오른쪽 제거
        mask[:, crop_right:] = 0

        # 페더링 (부드러운 경계)
        mask_float = mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (15, 15), 0)

        # 기존 알파 채널과 마스크 결합
        alpha_original = img_array[:, :, 3].astype(np.float32) / 255.0
        alpha_combined = alpha_original * mask_blurred
        img_array[:, :, 3] = (alpha_combined * 255).astype(np.uint8)

        # PIL Image로 변환
        result_image = Image.fromarray(img_array, 'RGBA')

        # 크롭 영역 계산
        crop_top = max(0, face_center_y - circle_radius - 10)
        crop_bottom = min(height, face_center_y + circle_radius + 10)

        result_image = result_image.crop((crop_left, crop_top, crop_right, crop_bottom))

        logger.info(f"[{filename}] Circular crop created from normalized (radius: {circle_radius}, width: {crop_right - crop_left})")

        return result_image

    def _scale_to_face_height(self, image, detection, target_face_height, filename="unknown"):
        """얼굴 높이를 기준으로 이미지를 스케일링하고 좌표를 재계산"""
        width, height = image.size
        fx, fy, fw, fh = detection['face_bbox']
        face_center_x, face_center_y = detection['face_center']

        scale_factor = target_face_height / max(fh, 1)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        scaled_detection = {
            'fx': int(fx * scale_factor),
            'fy': int(fy * scale_factor),
            'fw': int(fw * scale_factor),
            'fh': int(fh * scale_factor),
            'face_center_x': int(face_center_x * scale_factor),
            'face_center_y': int(face_center_y * scale_factor),
        }

        logger.info(f"[{filename}] Scaled image from {width}x{height} to {new_width}x{new_height} (scale: {scale_factor:.2f})")
        return scaled_image, scaled_detection

    def _create_final_upper_body(self, image, detection, filename="unknown"):
        """
        최종 버전 1: VS 화면용 어깨선 상반신
        - 얼굴 크기 표준화 (모든 이미지 동일한 얼굴 크기)
        - 상체 비율 표준화 (얼굴 기준 일정한 비율)
        - 일관된 크기로 출력 (600x640)

        Args:
            image: PIL Image (RGBA)
            detection: MediaPipe 감지 결과
            filename: 파일명

        Returns:
            PIL Image: VS용 상반신 이미지
        """
        logger.info(f"[{filename}] Creating final upper body (Version 1)...")

        # 1. 얼굴 크기 표준화: 모든 얼굴을 동일한 크기로 스케일링
        scaled_image, sd = self._scale_to_face_height(image, detection, 200, filename)
        scaled_fy, scaled_fw, scaled_fh = sd['fy'], sd['fw'], sd['fh']
        scaled_face_center_x = sd['face_center_x']

        # 2. 일정한 비율로 크롭 (얼굴 기준, 머리카락 포함)
        scaled_array = np.array(scaled_image)
        scaled_height, scaled_width = scaled_array.shape[:2]
        scaled_alpha = scaled_array[:, :, 3]

        # 실제 콘텐츠 최상단 확인 (머리카락 포함)
        content_rows = np.any(scaled_alpha > 10, axis=1)
        content_row_indices = np.where(content_rows)[0]

        if len(content_row_indices) > 0:
            actual_top = content_row_indices.min()
        else:
            actual_top = scaled_fy

        # 상하 크롭: 실제 머리카락 최상단부터 + 얼굴 아래 일정 비율 (어깨 포함)
        crop_top = max(0, actual_top - 10)  # 머리카락 최상단 + 약간 여유
        crop_bottom = min(scaled_height, scaled_fy + scaled_fh + int(scaled_fh * 0.5))  # 얼굴 아래 50%

        # 좌우 크롭: 얼굴 중심 기준으로 일정 범위만 (와이드하게 나오지 않도록)
        # 얼굴 너비의 1.8배를 최대 폭으로 제한
        max_half_width = int(scaled_fw * 0.9)  # 얼굴 중심에서 좌우 각각 얼굴 너비의 90%
        crop_left = max(0, scaled_face_center_x - max_half_width)
        crop_right = min(scaled_width, scaled_face_center_x + max_half_width)

        # 크롭 실행
        result_image = scaled_image.crop((crop_left, crop_top, crop_right, crop_bottom))

        # 3. 너비만 고정, 높이는 비율 유지 (여백 없음)
        if result_image.width != VERSION1_TARGET_WIDTH:
            scale = VERSION1_TARGET_WIDTH / result_image.width
            new_height = int(result_image.height * scale)
            result_image = result_image.resize((VERSION1_TARGET_WIDTH, new_height), Image.Resampling.LANCZOS)

        logger.info(f"[{filename}] Final upper body created: {result_image.size}")
        return result_image

    def _create_final_circular_crop_wide(self, image, detection, filename="unknown"):
        """
        최종 버전 2: 시상 내역용 얼굴 타원형 크롭
        - 얼굴 크기를 표준화하여 모든 이미지의 얼굴 크기를 일정하게 만듦
        - 타원형 마스크를 적용하여 배경 제거
        - 얼굴을 중심에 배치

        Args:
            image: PIL Image (RGBA) - 버전1 이미지 (상반신)
            detection: MediaPipe 감지 결과
            filename: 파일명

        Returns:
            PIL Image: 시상용 타원형 얼굴 이미지
        """
        logger.info(f"[{filename}] Creating elliptical face crop from version1...")

        # 1. 얼굴 크기 표준화: 모든 얼굴을 동일한 크기로 스케일링
        scaled_image, sd = self._scale_to_face_height(image, detection, 300, filename)
        scaled_fy, scaled_fw, scaled_fh = sd['fy'], sd['fw'], sd['fh']
        scaled_face_center_x = sd['face_center_x']
        scaled_face_center_y = sd['face_center_y']

        # 2. 타원형 마스크 생성
        # numpy 배열로 변환
        scaled_array = np.array(scaled_image)
        scaled_height, scaled_width = scaled_array.shape[:2]
        scaled_alpha = scaled_array[:, :, 3]

        # 실제 콘텐츠 영역 확인 (머리카락 포함)
        content_rows = np.any(scaled_alpha > 10, axis=1)
        content_row_indices = np.where(content_rows)[0]

        if len(content_row_indices) > 0:
            actual_top = content_row_indices.min()
        else:
            actual_top = scaled_fy

        # 타원의 크기 계산 (실제 콘텐츠 기준)
        # 가로 반지름: 얼굴 너비의 75% (귀 포함)
        ellipse_radius_x = int(scaled_fw * 0.75)

        # 세로 반지름: 가로의 1.5배 - 상체가 조금 더 포함되도록
        ellipse_radius_y = int(ellipse_radius_x * 1.5)

        # 타원 중심: 머리카락이 잘리지 않도록 조정
        # 머리카락 최상단부터 타원 중심까지의 거리가 세로 반지름보다 작아야 함
        head_top_to_center = ellipse_radius_y - int(scaled_fh * 0.6)  # 위쪽은 적게, 아래쪽은 많이
        ellipse_center_y = actual_top + head_top_to_center
        ellipse_center_x = scaled_face_center_x

        # 타원 중심이 얼굴보다 약간 아래로 (상체가 더 나오도록)
        min_center_y = scaled_face_center_y - int(scaled_fh * 0.2)
        max_center_y = scaled_face_center_y + int(scaled_fh * 0.4)
        ellipse_center_y = max(min_center_y, min(ellipse_center_y, max_center_y))

        # 타원형 마스크 생성
        mask = np.zeros((scaled_height, scaled_width), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (ellipse_center_x, ellipse_center_y),
            (ellipse_radius_x, ellipse_radius_y),
            0,  # 회전 각도
            0,  # 시작 각도
            360,  # 끝 각도
            255,  # 색상
            -1  # 채우기
        )

        # 페더링 (부드러운 경계)
        mask_float = mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (21, 21), 0)

        # 기존 알파 채널과 마스크 결합
        alpha_original = scaled_array[:, :, 3].astype(np.float32) / 255.0
        alpha_combined = alpha_original * mask_blurred
        scaled_array[:, :, 3] = (alpha_combined * 255).astype(np.uint8)

        result_image = Image.fromarray(scaled_array, 'RGBA')

        # 3. 타원 영역만 크롭
        crop_left = max(0, ellipse_center_x - ellipse_radius_x - 20)
        crop_right = min(scaled_width, ellipse_center_x + ellipse_radius_x + 20)
        crop_top = max(0, ellipse_center_y - ellipse_radius_y - 20)
        crop_bottom = min(scaled_height, ellipse_center_y + ellipse_radius_y + 20)

        result_image = result_image.crop((crop_left, crop_top, crop_right, crop_bottom))

        logger.info(f"[{filename}] Elliptical crop created: {result_image.size}, ellipse: ({ellipse_radius_x}x{ellipse_radius_y})")

        # 4. 최종 크기 조정 (너무 크거나 작지 않도록)
        max_width = 400
        max_height = 500

        if result_image.width > max_width or result_image.height > max_height:
            # 비율 유지하면서 크기 조정
            width_ratio = max_width / result_image.width
            height_ratio = max_height / result_image.height
            scale = min(width_ratio, height_ratio)

            final_width = int(result_image.width * scale)
            final_height = int(result_image.height * scale)
            result_image = result_image.resize((final_width, final_height), Image.Resampling.LANCZOS)
            logger.info(f"[{filename}] Final resize to: {result_image.size}")

        return result_image

    def process_image(self, input_path, output_path):
        """
        파일 경로 기반 이미지 처리 (네 가지 버전 저장)

        Args:
            input_path: 입력 이미지 경로
            output_path: 출력 이미지 경로 (확장자 제외한 base path)
        """
        import os
        filename = os.path.basename(input_path)

        # 이미지 열기
        with Image.open(input_path) as input_image:
            # 배경 제거 (두 가지 최종 버전)
            version1, version2 = self.remove_background(input_image, filename=filename)

            # 출력 경로 생성
            base_path = os.path.splitext(output_path)[0]
            output_path_1 = f"{base_path}_bg_removed.png"
            output_path_2 = f"{base_path}_bg_removed_for_award.png"

            # PNG로 저장 (투명도 유지)
            version1.save(output_path_1, format='PNG')
            logger.info(f"[{filename}] Version 1 (Upper body) saved to: {output_path_1}")

            version2.save(output_path_2, format='PNG')
            logger.info(f"[{filename}] Version 2 (Wide circular) saved to: {output_path_2}")