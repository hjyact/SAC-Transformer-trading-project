"""
utils/plot_style.py — matplotlib 한글 폰트 일괄 설정

Windows 기본은 "Malgun Gothic", macOS 는 "AppleGothic", Linux 는 "NanumGothic".
사용 가능한 폰트가 없으면 DejaVu Sans 로 폴백.

또한 폰트 변경 시 마이너스 부호가 □ 로 깨지는 문제를
`axes.unicode_minus=False` 로 차단한다.
"""

from __future__ import annotations

import platform
import logging
from typing import List

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_KOREAN_CANDIDATES_BY_OS = {
    "Windows": ["Malgun Gothic", "NanumGothic", "Gulim", "Batang"],
    "Darwin":  ["AppleGothic", "NanumGothic", "Apple SD Gothic Neo"],
    "Linux":   ["NanumGothic", "Noto Sans CJK KR", "UnDotum"],
}

_applied = False


def _available_fonts() -> set:
    return {f.name for f in fm.fontManager.ttflist}


def apply_korean_font(force: bool = False) -> str:
    """
    한글 폰트를 matplotlib rcParams 에 등록하고 적용된 폰트명을 반환.

    Parameters
    ----------
    force : True 면 이미 적용했더라도 다시 적용
    """
    global _applied
    if _applied and not force:
        return matplotlib.rcParams.get("font.family", "")

    candidates: List[str] = _KOREAN_CANDIDATES_BY_OS.get(
        platform.system(), _KOREAN_CANDIDATES_BY_OS["Linux"]
    )

    available = _available_fonts()
    chosen = next((c for c in candidates if c in available), None)

    if chosen is None:
        logger.warning(
            "한글 폰트를 찾을 수 없습니다 (%s). DejaVu Sans 로 대체 — "
            "한글이 □ 로 표시될 수 있습니다.", platform.system()
        )
        chosen = "DejaVu Sans"

    plt.rcParams["font.family"]       = chosen
    plt.rcParams["axes.unicode_minus"] = False   # '-' 부호 깨짐 방지
    _applied = True

    logger.info(f"matplotlib 폰트 적용: {chosen}")
    return chosen
