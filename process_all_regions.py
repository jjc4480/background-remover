#!/usr/bin/env python3
"""
지역별 디렉토리 포함 모든 이미지 처리
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from background_remover import BackgroundRemover
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 설정
    base_dir = Path('bg_remove_test_images')
    output_base = Path('bg_remove_test_output')
    supported_formats = {'.jpg', '.jpeg', '.png', '.webp'}

    # 모든 이미지 파일 찾기
    image_files = []
    for file in base_dir.rglob('*'):
        if file.is_file() and file.suffix.lower() in supported_formats:
            image_files.append(file)

    logger.info(f'총 {len(image_files)}개 이미지 발견')

    # 배경 제거 처리
    bg_remover = BackgroundRemover(model='birefnet-general')

    for i, img_file in enumerate(image_files, 1):
        try:
            # 상대 경로 계산
            rel_path = img_file.relative_to(base_dir)

            # 출력 경로 생성 (디렉토리 구조 유지)
            # process_image에서 _bg_removed와 _bg_removed_for_award를 추가하므로 여기서는 stem만 사용
            output_file = output_base / rel_path.parent / f'{rel_path.stem}.png'
            output_file.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f'[{i}/{len(image_files)}] Processing: {rel_path}')
            bg_remover.process_image(str(img_file), str(output_file))

        except Exception as e:
            logger.error(f'오류 발생 ({rel_path}): {e}')
            continue

    logger.info('모든 이미지 처리 완료!')

if __name__ == "__main__":
    main()
