"""프로젝트 공통 설정: 지하수 측정값의 방향을 한 곳에서 관리한다.

측정값(원본 '지하수위' 컬럼)의 기준은 명세에서 확인하지 못했다.
강수 뒤 값이 줄어드는 반응을 근거로 '지표(또는 관정 상단)에서 물까지의 깊이'로 가정한다 (docs/PLAN.md 3장).
나중에 해발 수위로 확인되면 아래 한 줄만 False로 바꾸면 모든 도구의 해석이 예전으로 돌아간다.
원본 값은 어느 쪽이든 바꾸지 않는다. 방향은 등급 판정과 설명 문장에만 쓴다.
"""

# True: 값이 클수록 물이 깊다(지하수가 낮다). False: 값이 클수록 지하수가 높다(해발 수위).
VALUE_IS_DEPTH = True


def water_sign():
    """측정값을 '물 높이 방향'으로 바꾸는 부호를 돌려준다 (깊이면 −1, 해발 수위면 +1)."""
    return -1 if VALUE_IS_DEPTH else 1


def value_label():
    """화면·결과에 쓸 측정값 이름을 돌려준다."""
    return "측정값(물까지의 깊이로 추정, m)" if VALUE_IS_DEPTH else "측정값(해발 수위로 추정, m)"


def water_move(value_change):
    """측정값 변화량(m)을 '물이 x m 차올랐습니다/내려갔습니다' 문장으로 바꾼다."""
    water = water_sign() * value_change
    if abs(water) < 0.005:
        return "물 높이가 거의 그대로였습니다"
    return f"물이 {abs(water):.2f} m {'차올랐습니다' if water > 0 else '내려갔습니다'}"


def water_expect(value_change):
    """예측 측정값 변화량(m)을 '물이 x m 차오를/내려갈 것으로 예상됩니다' 문장으로 바꾼다."""
    water = water_sign() * value_change
    if abs(water) < 0.005:
        return "물 높이가 거의 그대로일 것으로 예상됩니다"
    return f"물이 {abs(water):.2f} m {'차오를' if water > 0 else '내려갈'} 것으로 예상됩니다"


def water_compare(value_diff):
    """두 시점의 측정값 차이(m, 지금 − 비교 대상)를 '물이 x m 높음/낮음'으로 바꾼다."""
    water = water_sign() * value_diff
    if abs(water) < 0.005:
        return "물 높이가 거의 같음"
    return f"물이 {abs(water):.2f} m {'높음' if water > 0 else '낮음'}"


def axis_note():
    """그래프 y축 설명을 돌려준다 (깊이면 축을 뒤집어 위쪽이 물이 높은 쪽이 되게 그린다)."""
    return value_label() + ("\n(축 반전: 위로 갈수록 물이 높음)" if VALUE_IS_DEPTH else "")


def orient_axis(ax):
    """깊이 가정이면 y축을 뒤집어, 어느 가정이든 그래프의 위쪽이 물이 높은 쪽이 되게 한다."""
    if VALUE_IS_DEPTH:
        ax.invert_yaxis()
