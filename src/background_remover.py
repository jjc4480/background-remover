"""
배경 제거 핵심 로직 모듈
Lambda와 로컬 환경에서 공통으로 사용
"""
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove, new_session
import cv2
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)

# 상반신 표준화 설정 (증명사진 스타일)
STANDARD_OUTPUT_WIDTH = 600   # 표준 출력 너비
STANDARD_OUTPUT_HEIGHT = 800  # 표준 출력 높이
SHOULDER_TO_TOP_RATIO = 0.35  # 어깨에서 상단까지 비율 (머리 공간)


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
        이미지 배경 제거

        Args:
            input_image: PIL Image 객체
            filename: 파일명 (로깅용)

        Returns:
            PIL Image: 배경이 제거된 이미지 (RGBA)
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

        # 상반신 표준화 (증명사진 스타일)
        output_image = self._normalize_upper_body(output_image, filename=filename)

        return output_image

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

            # 2. 얼굴 크기가 이미지의 최소 5% 이상이어야 함 (너무 작으면 오탐지)
            face_area = fw * fh
            image_area = img_width * img_height
            if face_area < image_area * 0.05:
                logger.info(f"[{filename}] ❌ Rejected face at ({fx}, {fy}, {fw}, {fh}) - too small ({face_area}/{image_area} = {face_area/image_area*100:.1f}%)")
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

        logger.info(f"Normalized from {image.size} to {canvas.size} (scale: {scale_factor:.2f})")

        return canvas

    def process_image(self, input_path, output_path):
        """
        파일 경로 기반 이미지 처리

        Args:
            input_path: 입력 이미지 경로
            output_path: 출력 이미지 경로
        """
        import os
        filename = os.path.basename(input_path)

        # 이미지 열기
        with Image.open(input_path) as input_image:
            # 배경 제거
            output_image = self.remove_background(input_image, filename=filename)

            # PNG로 저장 (투명도 유지)
            output_image.save(output_path, format='PNG')
            logger.info(f"[{filename}] Saved to: {output_path}")