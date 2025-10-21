"""
배경 제거 핵심 로직 모듈
Lambda와 로컬 환경에서 공통으로 사용
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove, new_session
import cv2
from scipy.ndimage import gaussian_filter
from skimage import exposure
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

# 상반신 표준화 설정 (증명사진 스타일)
STANDARD_OUTPUT_WIDTH = 600   # 표준 출력 너비
STANDARD_OUTPUT_HEIGHT = 800  # 표준 출력 높이
SHOULDER_TO_TOP_RATIO = 0.35  # 어깨에서 상단까지 비율 (머리 공간)

# 최종 버전별 고정 너비
VERSION1_TARGET_WIDTH = 480   # 버전1: 상반신 고정 너비
VERSION2_TARGET_WIDTH = 300   # 버전2: 원형 (머리카락 포함) 고정 너비


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
            # MediaPipe 실패 시 Haar Cascade로 fallback
            logger.warning(f"[{filename}] MediaPipe not available, using Haar Cascade fallback")
            version1 = self._normalize_upper_body(output_image, filename=filename)
            version2 = self._create_circular_face_crop_from_normalized(version1.copy(), filename=filename)
            return version1, version2

        # MediaPipe 감지
        detection = detector.detect_face_and_body(output_image, filename=filename)

        if detection is None:
            # 감지 실패 시 Haar Cascade로 fallback
            logger.warning(f"[{filename}] MediaPipe detection failed, using Haar Cascade fallback")
            version1 = self._normalize_upper_body(output_image, filename=filename)
            version2 = self._create_circular_face_crop_from_normalized(version1.copy(), filename=filename)
            return version1, version2

        # MediaPipe 기반 최종 버전 생성
        version1 = self._create_final_upper_body(output_image.copy(), detection, filename=filename)
        version2 = self._create_final_circular_crop_wide(version1.copy(), detection, filename=filename)

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

        # 얼굴 검출
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            logger.warning(f"[{filename}] No face detected")
            return None

        # 유효한 얼굴 필터링
        valid_faces = []
        img_height = image.height
        img_width = image.width

        for (fx, fy, fw, fh) in faces:
            # 1. 얼굴이 이미지 상단 70% 영역에 있어야 함 (하단은 잘못된 검출 가능성 높음)
            if fy + fh > img_height * 0.7:
                logger.info(f"[{filename}] ❌ Rejected face at ({fx}, {fy}, {fw}, {fh}) - too low (bottom > 70%)")
                continue

            # 2. 얼굴 크기 검증 (해상도 적응형)
            face_area = fw * fh
            image_area = img_width * img_height

            # 절대 크기: 최소 50x50 픽셀
            if fw < 50 or fh < 50:
                logger.info(f"[{filename}] ❌ Rejected face at ({fx}, {fy}, {fw}, {fh}) - too small (min 50x50)")
                continue

            # 상대 크기: 해상도에 따라 적응형
            max_dimension = max(img_width, img_height)
            if max_dimension < 500:
                min_ratio = 0.05  # 저해상도: 5%
            elif max_dimension < 1500:
                min_ratio = 0.03  # 중해상도: 3%
            else:
                min_ratio = 0.01  # 고해상도: 1%

            if face_area < image_area * min_ratio:
                logger.info(f"[{filename}] ❌ Rejected face at ({fx}, {fy}, {fw}, {fh}) - too small ({face_area}/{image_area} = {face_area/image_area*100:.1f}%, min: {min_ratio*100:.1f}%)")
                continue

            # 3. 얼굴 비율이 정상 범위여야 함 (너무 길거나 넓으면 오탐지)
            aspect_ratio = fw / fh
            if aspect_ratio < 0.6 or aspect_ratio > 1.5:
                logger.info(f"[{filename}] ❌ Rejected face at ({fx}, {fy}, {fw}, {fh}) - abnormal aspect ratio ({aspect_ratio:.2f})")
                continue

            valid_faces.append((fx, fy, fw, fh))

        if len(valid_faces) == 0:
            logger.warning(f"[{filename}] ⚠️  No valid face detected after filtering (found {len(faces)} faces but all rejected)")
            return None

        # 가장 상단에 있고 큰 얼굴 선택 (상단 우선, 크기 차선)
        face = min(valid_faces, key=lambda f: (f[1], -f[2] * f[3]))
        fx, fy, fw, fh = face

        # 어깨선 추정: 얼굴 하단 + 얼굴 높이의 30-40%
        shoulder_y = fy + fh + int(fh * 0.35)

        logger.info(f"[{filename}] ✅ Valid face detected at ({fx}, {fy}, {fw}, {fh}), shoulder_y estimated at {shoulder_y}")

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

    def _create_circular_face_crop(self, image, filename="unknown"):
        """
        얼굴 중심 원형 크롭 생성 (귀 기준 양옆 제거)

        Args:
            image: PIL Image (RGBA)
            filename: 파일명 (로깅용)

        Returns:
            PIL Image: 원형 크롭된 이미지
        """
        logger.info(f"[{filename}] Creating circular face crop...")

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

        # 좌우 크롭 범위: 얼굴 너비 기준으로 귀까지만 (얼굴 너비 * 1.1)
        crop_width_half = int(fw * 0.55)  # 얼굴 중심에서 양옆으로 각각 얼굴너비의 55%
        crop_left = max(0, face_center_x - crop_width_half)
        crop_right = min(img_array.shape[1], face_center_x + crop_width_half)

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

        logger.info(f"[{filename}] Circular crop with vertical edges created (radius: {circle_radius}, width: {crop_right - crop_left})")

        return result_image

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

    def _create_mediapipe_versions(self, image, filename="unknown"):
        """
        MediaPipe 기반 버전 3, 4 생성

        Args:
            image: PIL Image (RGBA) - 배경 제거된 원본 이미지
            filename: 파일명 (로깅용)

        Returns:
            tuple: (version3, version4)
                - version3: 어깨선까지 포함된 상반신 (MediaPipe)
                - version4: 얼굴 중심 원형 크롭 (MediaPipe)
        """
        logger.info(f"[{filename}] Creating MediaPipe versions...")

        # MediaPipe 감지기 가져오기
        detector = get_mediapipe_detector()

        if detector is None:
            logger.warning(f"[{filename}] MediaPipe not available, returning empty images")
            # 빈 이미지 반환
            empty = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
            return empty, empty

        # 얼굴 및 신체 감지
        detection = detector.detect_face_and_body(image, filename=filename)

        if detection is None:
            logger.warning(f"[{filename}] MediaPipe detection failed, returning original")
            return image.copy(), image.copy()

        # Version 3: 상반신 표준화 (MediaPipe)
        version3 = self._normalize_upper_body_mediapipe(image.copy(), detection, filename=filename)

        # Version 4: 버전3을 기반으로 얼굴 중심 원형 크롭 (MediaPipe)
        version4 = self._create_circular_face_crop_mediapipe(version3.copy(), detection, filename=filename)

        return version3, version4

    def _normalize_upper_body_mediapipe(self, image, detection, filename="unknown"):
        """
        MediaPipe 감지 결과로 상반신 표준화
        핵심: 손 포즈를 감지해서 제외하고, 얼굴을 중심에 정확히 배치

        Args:
            image: PIL Image (RGBA)
            detection: MediaPipe 감지 결과
            filename: 파일명

        Returns:
            PIL Image: 정규화된 상반신 이미지
        """
        logger.info(f"[{filename}] MediaPipe upper body normalization...")

        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # 얼굴 정보
        fx, fy, fw, fh = detection['face_bbox']
        face_center_x, face_center_y = detection['face_center']

        # 어깨 위치 감지
        shoulders = detection.get('shoulders', [])

        if len(shoulders) >= 2:
            shoulder_y = max(shoulders[0][1], shoulders[1][1])
            logger.info(f"[{filename}] Both shoulders detected at y={shoulder_y}")
        elif len(shoulders) == 1:
            shoulder_y = shoulders[0][1]
            logger.info(f"[{filename}] One shoulder detected at y={shoulder_y}")
        else:
            shoulder_y = fy + int(fh * 2.5)
            logger.info(f"[{filename}] No shoulders detected, estimated at y={shoulder_y}")

        # 상하 크롭: 머리 위부터 어깨 아래까지
        crop_top = max(0, fy - int(fh * 0.8))  # 얼굴 위로 여유있게
        crop_bottom = min(height, shoulder_y + int(fh * 0.3))

        # 좌우 크롭: 얼굴과 어깨만 기준 (손 포즈 제외)
        # 1. 얼굴+어깨 영역에서만 픽셀 감지
        alpha = img_array[:, :, 3]

        # 얼굴부터 어깨까지 영역만 검사
        body_region_alpha = alpha[fy:shoulder_y, :]

        # 손 위치 확인
        hands = detection.get('hands', [])
        hand_x_positions = []
        for hand_x, hand_y in hands:
            # 어깨 위쪽에 있는 손만 (포즈)
            if hand_y < shoulder_y:
                hand_x_positions.append(hand_x)
                logger.info(f"[{filename}] Hand pose detected at x={hand_x}, y={hand_y} - will exclude from crop")

        # 몸 영역 픽셀 감지 (손 영역 제외)
        body_pixels = np.any(body_region_alpha > 10, axis=0)

        # 손 영역 마스킹 (손 주변 픽셀 제외)
        for hand_x in hand_x_positions:
            # 손 주변 ±fw*0.5 범위 제외
            exclude_range = int(fw * 0.5)
            exclude_left = max(0, hand_x - exclude_range)
            exclude_right = min(width, hand_x + exclude_range)
            body_pixels[exclude_left:exclude_right] = False

        body_indices = np.where(body_pixels)[0]

        if len(body_indices) > 0:
            body_left = body_indices.min()
            body_right = body_indices.max()

            # 얼굴 중심 기준으로 대칭 크롭
            left_distance = face_center_x - body_left
            right_distance = body_right - face_center_x

            # 대칭을 위해 큰 쪽 사용 (단, 최대 얼굴 너비의 1.8배로 제한)
            max_distance = min(max(left_distance, right_distance), int(fw * 1.8))

            crop_left = max(0, face_center_x - max_distance)
            crop_right = min(width, face_center_x + max_distance)

            logger.info(f"[{filename}] Body crop (excluding hands): left_dist={left_distance}, right_dist={right_distance}, max={max_distance}")
        else:
            # fallback: 얼굴만 기준
            crop_left = max(0, face_center_x - int(fw * 1.5))
            crop_right = min(width, face_center_x + int(fw * 1.5))
            logger.info(f"[{filename}] Fallback to face-only crop")

        # 크롭
        cropped_image = image.crop((crop_left, crop_top, crop_right, crop_bottom))

        # 최종 여백 제거
        cropped_image = self._crop_to_content(cropped_image)

        logger.info(f"[{filename}] MediaPipe upper body normalized to {cropped_image.size}")
        return cropped_image

    def _create_circular_face_crop_mediapipe(self, image, detection, filename="unknown"):
        """
        MediaPipe 감지 결과로 얼굴 중심 원형 크롭 생성
        핵심: 손 포즈를 제외하고 얼굴을 중심에 배치, 귀 기준 좌우 크롭

        Args:
            image: PIL Image (RGBA) - 정규화된 이미지
            detection: MediaPipe 감지 결과
            filename: 파일명

        Returns:
            PIL Image: 원형 크롭된 이미지
        """
        logger.info(f"[{filename}] MediaPipe circular face crop...")

        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # 얼굴 재감지 (정규화된 이미지에서)
        detector = get_mediapipe_detector()
        if detector:
            new_detection = detector.detect_face_and_body(image, filename=filename)
            if new_detection:
                detection = new_detection

        fx, fy, fw, fh = detection['face_bbox']
        face_center_x, face_center_y = detection['face_center']

        # 귀 위치 사용
        left_ear = detection.get('left_ear')
        right_ear = detection.get('right_ear')

        # 원형 반지름
        circle_radius = int(max(fw, fh) * 1.15)

        # 좌우 크롭: 귀 기준 + 머리카락 여유 (손 포즈는 무시)
        if left_ear and right_ear:
            # 귀 간격 기준 + 여유
            ear_width = right_ear[0] - left_ear[0]
            crop_width_half = int(ear_width * 0.65)  # 0.55 -> 0.65로 증가
            logger.info(f"[{filename}] Using ear positions for crop width: {crop_width_half}")
        else:
            # 얼굴 너비 기준 + 여유
            crop_width_half = int(fw * 0.75)  # 0.65 -> 0.75로 증가
            logger.info(f"[{filename}] No ears detected, using face width: {crop_width_half}")

        # 대칭 크롭 (얼굴 중심 기준)
        crop_left = max(0, face_center_x - crop_width_half)
        crop_right = min(width, face_center_x + crop_width_half)

        # 원형 마스크
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (face_center_x, face_center_y), circle_radius, 255, -1)

        # 좌우 제거 (수직선)
        mask[:, :crop_left] = 0
        mask[:, crop_right:] = 0

        # 페더링
        mask_float = mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (15, 15), 0)

        # 알파 채널 결합
        alpha_original = img_array[:, :, 3].astype(np.float32) / 255.0
        alpha_combined = alpha_original * mask_blurred
        img_array[:, :, 3] = (alpha_combined * 255).astype(np.uint8)

        result_image = Image.fromarray(img_array, 'RGBA')

        # 크롭
        crop_top = max(0, face_center_y - circle_radius - 10)
        crop_bottom = min(height, face_center_y + circle_radius + 10)
        result_image = result_image.crop((crop_left, crop_top, crop_right, crop_bottom))

        logger.info(f"[{filename}] MediaPipe circular crop created (radius: {circle_radius}, width: {crop_right - crop_left})")

        return result_image

    def _create_final_upper_body(self, image, detection, filename="unknown"):
        """
        최종 버전 1: VS 화면용 어깨선 상반신
        - MediaPipe로 정확한 어깨/얼굴 감지
        - 손 포즈 제외
        - 일관된 크기로 리사이징 (600x800 표준)
        - 얼굴이 중심에 배치

        Args:
            image: PIL Image (RGBA)
            detection: MediaPipe 감지 결과
            filename: 파일명

        Returns:
            PIL Image: VS용 상반신 이미지
        """
        logger.info(f"[{filename}] Creating final upper body (Version 1)...")

        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # 얼굴 정보
        fx, fy, fw, fh = detection['face_bbox']
        face_center_x, face_center_y = detection['face_center']

        # 어깨 위치
        shoulders = detection.get('shoulders', [])
        if len(shoulders) >= 2:
            shoulder_y = max(shoulders[0][1], shoulders[1][1])
        elif len(shoulders) == 1:
            shoulder_y = shoulders[0][1]
        else:
            shoulder_y = fy + int(fh * 2.5)

        # 상하 크롭
        crop_top = max(0, fy - int(fh * 0.8))
        crop_bottom = min(height, shoulder_y + int(fh * 0.3))

        # 좌우 크롭 (손 포즈 제외)
        alpha = img_array[:, :, 3]
        body_region_alpha = alpha[fy:shoulder_y, :]

        # 손 위치 제외
        hands = detection.get('hands', [])
        body_pixels = np.any(body_region_alpha > 10, axis=0)

        for hand_x, hand_y in hands:
            if hand_y < shoulder_y:
                exclude_range = int(fw * 0.5)
                exclude_left = max(0, hand_x - exclude_range)
                exclude_right = min(width, hand_x + exclude_range)
                body_pixels[exclude_left:exclude_right] = False

        body_indices = np.where(body_pixels)[0]

        if len(body_indices) > 0:
            body_left = body_indices.min()
            body_right = body_indices.max()
            left_distance = face_center_x - body_left
            right_distance = body_right - face_center_x
            max_distance = min(max(left_distance, right_distance), int(fw * 1.8))
            crop_left = max(0, face_center_x - max_distance)
            crop_right = min(width, face_center_x + max_distance)
        else:
            crop_left = max(0, face_center_x - int(fw * 1.5))
            crop_right = min(width, face_center_x + int(fw * 1.5))

        # 크롭
        cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))

        # 표준 크기로 리사이징 (STANDARD_OUTPUT_WIDTH x STANDARD_OUTPUT_HEIGHT)
        # 얼굴이 출력 너비의 37% 차지하도록
        target_face_width = int(STANDARD_OUTPUT_WIDTH * 0.37)
        scale_factor = target_face_width / max(fw, 1)

        new_width = int(cropped.width * scale_factor)
        new_height = int(cropped.height * scale_factor)
        scaled = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 캔버스에 배치
        canvas_height = min(STANDARD_OUTPUT_HEIGHT, max(new_height + 100, int(STANDARD_OUTPUT_HEIGHT * 0.10) + new_height))
        canvas = Image.new('RGBA', (STANDARD_OUTPUT_WIDTH, canvas_height), (0, 0, 0, 0))

        paste_x = (STANDARD_OUTPUT_WIDTH - new_width) // 2
        paste_y = int(canvas_height * 0.08)

        # 높이/너비 오버플로우 처리
        if paste_y + new_height > canvas_height:
            scaled = scaled.crop((0, 0, new_width, canvas_height - paste_y))
        if new_width > STANDARD_OUTPUT_WIDTH:
            paste_x = 0
            scaled = scaled.crop(((new_width - STANDARD_OUTPUT_WIDTH) // 2, 0,
                                 (new_width + STANDARD_OUTPUT_WIDTH) // 2, scaled.height))

        canvas.paste(scaled, (paste_x, paste_y), scaled)

        # 최종 여백 제거
        canvas = self._crop_to_content(canvas)

        # 고정 너비로 리사이징 (비율 유지)
        if canvas.width != VERSION1_TARGET_WIDTH:
            scale = VERSION1_TARGET_WIDTH / canvas.width
            new_height = int(canvas.height * scale)
            canvas = canvas.resize((VERSION1_TARGET_WIDTH, new_height), Image.Resampling.LANCZOS)
            logger.info(f"[{filename}] Resized to fixed width: {canvas.size}")

        logger.info(f"[{filename}] Final upper body created: {canvas.size}")
        return canvas

    def _create_final_circular_crop(self, image, detection, filename="unknown"):
        """
        최종 버전 2: 시상 내역용 얼굴 원형 크롭
        - MediaPipe로 정확한 귀 위치 감지
        - 머리카락이 긴 경우 머리카락 포함
        - 손 포즈 제외
        - 양옆 수직 제거
        - 얼굴이 중심에 배치

        Args:
            image: PIL Image (RGBA) - 버전1 이미지
            detection: MediaPipe 감지 결과
            filename: 파일명

        Returns:
            PIL Image: 시상용 원형 이미지
        """
        logger.info(f"[{filename}] Creating final circular crop (Version 2)...")

        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # 얼굴 재감지 (버전1 이미지에서)
        detector = get_mediapipe_detector()
        if detector:
            new_detection = detector.detect_face_and_body(image, filename=filename)
            if new_detection:
                detection = new_detection

        fx, fy, fw, fh = detection['face_bbox']
        face_center_x, face_center_y = detection['face_center']

        # 원형 반지름
        circle_radius = int(max(fw, fh) * 1.15)

        # 귀 위치 기반 좌우 크롭
        left_ear = detection.get('left_ear')
        right_ear = detection.get('right_ear')

        if left_ear and right_ear:
            ear_width = right_ear[0] - left_ear[0]
            crop_width_half = int(ear_width * 0.65)
        else:
            crop_width_half = int(fw * 0.75)

        # 대칭 크롭
        crop_left = max(0, face_center_x - crop_width_half)
        crop_right = min(width, face_center_x + crop_width_half)

        # 원형 마스크
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (face_center_x, face_center_y), circle_radius, 255, -1)

        # 양옆 수직 제거
        mask[:, :crop_left] = 0
        mask[:, crop_right:] = 0

        # 페더링
        mask_float = mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (15, 15), 0)

        # 알파 채널 결합
        alpha_original = img_array[:, :, 3].astype(np.float32) / 255.0
        alpha_combined = alpha_original * mask_blurred
        img_array[:, :, 3] = (alpha_combined * 255).astype(np.uint8)

        result_image = Image.fromarray(img_array, 'RGBA')

        # 크롭
        crop_top = max(0, face_center_y - circle_radius - 10)
        crop_bottom = min(height, face_center_y + circle_radius + 10)
        result_image = result_image.crop((crop_left, crop_top, crop_right, crop_bottom))

        # 고정 너비로 리사이징 (비율 유지)
        if result_image.width != VERSION2_TARGET_WIDTH:
            scale = VERSION2_TARGET_WIDTH / result_image.width
            new_height = int(result_image.height * scale)
            result_image = result_image.resize((VERSION2_TARGET_WIDTH, new_height), Image.Resampling.LANCZOS)
            logger.info(f"[{filename}] Resized to fixed width: {result_image.size}")

        logger.info(f"[{filename}] Final circular crop created: {result_image.size}")
        return result_image

    def _create_final_circular_crop_wide(self, image, detection, filename="unknown"):
        """
        최종 버전 3: 시상 내역용 얼굴 원형 크롭 (넓은 버전)
        - 머리카락이 잘리지 않도록 더 넓게 크롭
        - MediaPipe로 정확한 감지
        - 손 포즈 제외
        - 양옆 수직 제거
        - 얼굴이 중심에 배치

        Args:
            image: PIL Image (RGBA) - 버전1 이미지
            detection: MediaPipe 감지 결과
            filename: 파일명

        Returns:
            PIL Image: 시상용 원형 이미지 (넓은 버전)
        """
        logger.info(f"[{filename}] Creating final wide circular crop (Version 3)...")

        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # 얼굴 재감지 (버전1 이미지에서)
        detector = get_mediapipe_detector()
        if detector:
            new_detection = detector.detect_face_and_body(image, filename=filename)
            if new_detection:
                detection = new_detection

        fx, fy, fw, fh = detection['face_bbox']
        face_center_x, face_center_y = detection['face_center']

        # 원형 반지름 (더 크게)
        circle_radius = int(max(fw, fh) * 1.3)  # 1.15 -> 1.3

        # 좌우 크롭: 실제 머리카락 범위 감지
        alpha = img_array[:, :, 3]

        # 머리 영역 (얼굴 위로 더 넓게)
        head_top = max(0, fy - int(fh * 1.2))
        head_bottom = min(height, fy + int(fh * 0.5))
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

            # 대칭을 위해 큰 쪽 기준 (제한 없음 - 머리카락 전부 포함)
            crop_width_half = max(left_distance, right_distance)

            # 최소 크기는 귀 기준으로
            left_ear = detection.get('left_ear')
            right_ear = detection.get('right_ear')

            if left_ear and right_ear:
                ear_width = right_ear[0] - left_ear[0]
                min_width_half = int(ear_width * 0.65)
                crop_width_half = max(crop_width_half, min_width_half)

            logger.info(f"[{filename}] Wide crop includes full hair: width_half={crop_width_half}")
        else:
            # fallback: 넓게
            crop_width_half = int(fw * 0.9)
            logger.info(f"[{filename}] Using wide fallback width: {crop_width_half}")

        # 대칭 크롭
        crop_left = max(0, face_center_x - crop_width_half)
        crop_right = min(width, face_center_x + crop_width_half)

        # 원형 마스크
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (face_center_x, face_center_y), circle_radius, 255, -1)

        # 양옆 수직 제거
        mask[:, :crop_left] = 0
        mask[:, crop_right:] = 0

        # 페더링
        mask_float = mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (15, 15), 0)

        # 알파 채널 결합
        alpha_original = img_array[:, :, 3].astype(np.float32) / 255.0
        alpha_combined = alpha_original * mask_blurred
        img_array[:, :, 3] = (alpha_combined * 255).astype(np.uint8)

        result_image = Image.fromarray(img_array, 'RGBA')

        # 크롭
        crop_top = max(0, face_center_y - circle_radius - 10)
        crop_bottom = min(height, face_center_y + circle_radius + 10)
        result_image = result_image.crop((crop_left, crop_top, crop_right, crop_bottom))

        # 고정 너비로 리사이징 (비율 유지)
        if result_image.width != VERSION2_TARGET_WIDTH:
            scale = VERSION2_TARGET_WIDTH / result_image.width
            new_height = int(result_image.height * scale)
            result_image = result_image.resize((VERSION2_TARGET_WIDTH, new_height), Image.Resampling.LANCZOS)
            logger.info(f"[{filename}] Resized to fixed width: {result_image.size}")

        logger.info(f"[{filename}] Final wide circular crop created: {result_image.size}")
        return result_image

    def _enhance_image_quality(self, image, filename="unknown"):
        """
        이미지 품질 향상 (노이즈 제거 + 자동 밝기 + 2x 업스케일)

        Args:
            image: PIL Image (RGBA)
            filename: 파일명 (로깅용)

        Returns:
            PIL Image: 품질이 향상된 이미지
        """
        logger.info(f"[{filename}] Enhancing image quality...")

        original_size = image.size

        # 알파 채널 분리
        if image.mode == 'RGBA':
            alpha = image.split()[3]
            rgb_image = image.convert('RGB')
        else:
            alpha = None
            rgb_image = image

        # numpy 배열로 변환
        img_array = np.array(rgb_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 1. 노이즈 제거 (Non-local Means Denoising - 고품질)
        denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)

        # 2. 자연스러운 밝기 자동 조절
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 평균 밝기 확인
        avg_brightness = np.mean(l)

        # 어두운 이미지만 보정 (평균 밝기 < 120)
        if avg_brightness < 120:
            # 매우 약한 CLAHE (자연스럽게)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            # 원본과 블렌딩 (70% 원본 + 30% 보정)
            l = cv2.addWeighted(l, 0.7, l_enhanced, 0.3, 0)
            logger.info(f"[{filename}] Brightness adjusted (avg: {avg_brightness:.1f})")

        lab_adjusted = cv2.merge([l, a, b])
        img_adjusted = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

        # RGB로 변환
        img_rgb = cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2RGB)
        enhanced = Image.fromarray(img_rgb)

        # 알파 채널 복원
        if alpha is not None:
            enhanced = enhanced.convert('RGBA')
            enhanced.putalpha(alpha)

        # 3. 2배 업스케일 (LANCZOS)
        new_width = original_size[0] * 2
        new_height = original_size[1] * 2
        upscaled = enhanced.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 4. 미세한 선명도 향상
        upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=3))

        logger.info(f"[{filename}] Enhanced from {original_size} to {upscaled.size}")
        return upscaled

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