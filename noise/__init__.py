import numpy as np
from typing import Tuple, Union

Coord = Union[Tuple[float, float], np.ndarray]

def add_jitter_to_segment(p1: Coord,
                          sigma: float = 0.5,
                          random_state: np.random.Generator | None = None
                          ) -> np.ndarray:
    """
    给线段两端点添加细微噪声。

    Parameters
    ----------
    p1 : (x, y)
        原始坐标。可为 tuple/list/np.ndarray，元素必须能转成 float。
    sigma : float, default 0.5
        噪声标准差，单位与坐标相同（像素或实数坐标）。
    random_state : np.random.Generator | None
        指定随机数生成器，便于可重复实验。默认为 None 使用全局随机状态。

    Returns
    -------
    p1_noisy : np.ndarray
        添加噪声后的坐标。
    """
    rng = random_state if random_state is not None else np.random.default_rng()

    p1 = np.asarray(p1, dtype=float)

    noise1 = rng.normal(loc=0.0, scale=sigma, size=2)

    p1_noisy = p1 + noise1
    return p1_noisy


# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    orig_p1 = (100, 150)

    p1_j = add_jitter_to_segment(orig_p1, sigma=0.8)
    print("原始:", orig_p1)
    print("加噪:", tuple(p1_j))
