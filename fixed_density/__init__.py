"""Fixed Density v1 —— osu!mania 4K 难度算法。

顶层只暴露最常用的求值入口（依赖 numpy，不引入 matplotlib / scipy）：

    from fixed_density import evaluate_osu
    print(evaluate_osu("x.osu"))          # {'rc_value': ..., 'ln_value': ..., 'n': ..., 'k': ...}

子模块：
    fixed_density.core           算法核心（参数注册表 / object_sequence / 映射曲线）
    fixed_density.value          端到端求值（.osu 或 (t,l) 列表 → value）
    fixed_density.reader         .osu → 逐轨 (t, l) 列表
    fixed_density.benchmark      benchmark 运行 / 拟合 / 统计 / HTML 报告（CLI）
    fixed_density.ln_benchmark   LN 参考难度 benchmark 报告（CLI）
    fixed_density.tune           全量参数调优（CLI）
    fixed_density.extract        benchmark 预处理（.osu → 物件元组 JSON）
"""

from .value import evaluate, evaluate_osu, load_final_model

__all__ = ["evaluate", "evaluate_osu", "load_final_model"]
__version__ = "1.0.0"
