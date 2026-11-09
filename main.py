"""
Лабораторная работа №3.
Вариант 4, трек A: многокритериальная оптимизация режима отопления помещения.
Критерии: минимум энергозатрат и минимум отклонения температуры от комфортной.
Реализован NSGA-II с нуля: недоминируемая сортировка, расстояние до скопления,
внешний архив, SBX-кроссовер и полиномиальная мутация.

Студент: Якимов Максим Андреевич
Группа: Б24-517, номер в группе 30.

Запуск:
    python main.py
Результаты:
    lab3_out/lab3_pareto.csv
    lab3_out/lab3_runs.csv
    lab3_out/lab3_hv.csv
    lab3_out/lab3_weighted.csv
    lab3_out/lab3_representatives.csv
    lab3_out/lab3_pareto.png
    lab3_out/lab3_hv.png
    lab3_out/lab3_dashboard.png
    lab3_out/lab3_block_schema.png
    lab3_out/lab3_schedule.png
"""

import os
import json
import csv
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Параметры индивидуального варианта и модели
# ---------------------------------------------------------------------------
BASE_SEED = 42
N_RUNS = 20
HOURS = 24
DT = 0.25                         # шаг интегрирования, ч
STEPS = int(HOURS / DT)           # 96 шагов
TIME = np.arange(STEPS) * DT

# Тепловая модель помещения (учебная, одно-зонная)
C_ROOM = 1.8                      # теплоёмкость, кВт·ч/К
UA = 0.22                         # коэффициент теплопотерь, кВт/К
P_MAX = 6.0                       # максимальная мощность отопления, кВт
KP = 3.0                          # коэффициент пропорционального регулятора, кВт/К
T_INIT = 20.0                     # начальная температура, °C

# Комфорт: днём (07:00–23:00) 20–23 °C, ночью 17–22 °C
OCCUPIED = ((TIME >= 7.0) & (TIME < 23.0)).astype(float)
T_LOW = np.where(OCCUPIED > 0, 20.0, 17.0)
T_HIGH = np.where(OCCUPIED > 0, 23.0, 22.0)
COMFORT_WEIGHT = np.where(OCCUPIED > 0, 1.0, 0.35)

# Уличная температура: от −4 °C ночью до +10 °C днём
T_OUT = 3.0 + 7.0 * np.sin(2.0 * np.pi * (TIME - 9.0) / 24.0)

# Границы генома: целевая температура по часам
LOWER, UPPER = 14.0, 24.0

# Опорная точка для гиперобъёма (фиксированная, чтобы сравнивать методы)
REF_POINT = np.array([120.0, 200.0])


# ---------------------------------------------------------------------------
# Модель и целевые функции
# ---------------------------------------------------------------------------
def simulate(pop: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Батчевая симуляция теплового режима.

    pop : (N, 24) — целевая температура на каждый час.
    Возвращает energy (N,), discomfort (N,), temperatures (N, 96), power (N, 96).
    """
    pop = np.asarray(pop, dtype=float)
    n = pop.shape[0]
    hour_idx = np.minimum((TIME // 1.0).astype(int), HOURS - 1)
    target = pop[:, hour_idx]                    # (N, 96)

    temp = np.full(n, T_INIT, dtype=float)
    energy = np.zeros(n, dtype=float)
    discomfort = np.zeros(n, dtype=float)
    temps = np.empty((n, STEPS), dtype=float)
    powers = np.empty((n, STEPS), dtype=float)

    for k in range(STEPS):
        power = np.clip(KP * (target[:, k] - temp), 0.0, P_MAX)
        temp = temp + DT / C_ROOM * (UA * (T_OUT[k] - temp) + power)
        powers[:, k] = power
        temps[:, k] = temp
        energy += power * DT
        deviation = (np.maximum(0.0, T_LOW[k] - temp)
                     + np.maximum(0.0, temp - T_HIGH[k]))
        discomfort += COMFORT_WEIGHT[k] * deviation * DT

    # Мягкий штраф за технологические нарушения (защита труб/перегрев).
    violation = (np.maximum(0.0, 8.0 - temps).sum(axis=1)
                 + np.maximum(0.0, temps - 30.0).sum(axis=1))
    energy = energy + 20.0 * violation
    discomfort = discomfort + 100.0 * violation
    return energy, discomfort, temps, powers


def evaluate(pop: np.ndarray) -> np.ndarray:
    e, d, _, _ = simulate(pop)
    return np.column_stack([e, d])


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(fitness: np.ndarray) -> List[np.ndarray]:
    n = len(fitness)
    dominated_by = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(p + 1, n):
            if dominates(fitness[p], fitness[q]):
                dominated_by[p].append(q)
                domination_count[q] += 1
            elif dominates(fitness[q], fitness[p]):
                dominated_by[q].append(p)
                domination_count[p] += 1

    current = [i for i in range(n) if domination_count[i] == 0]
    fronts[0] = current
    while current:
        nxt = []
        for p in current:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    nxt.append(q)
        if nxt:
            fronts.append(nxt)
        current = nxt
    return [np.array(f, dtype=int) for f in fronts if len(f) > 0]


def crowding_distance(front_fitness: np.ndarray) -> np.ndarray:
    """Расстояние до скопления для фронта (N, m)."""
    n, m = front_fitness.shape
    dist = np.zeros(n, dtype=float)
    if n <= 2:
        dist[:] = np.inf
        return dist
    for k in range(m):
        order = np.argsort(front_fitness[:, k])
        vals = front_fitness[order, k]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = vals[-1] - vals[0]
        if span <= 1e-12:
            continue
        for i in range(1, n - 1):
            dist[order[i]] += (vals[i + 1] - vals[i - 1]) / span
    return dist


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator,
                  eta: float = 15.0, pc: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    if rng.random() >= pc:
        return p1.copy(), p2.copy()
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5,
                    (2.0 * u) ** (1.0 / (eta + 1.0)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    return np.clip(c1, LOWER, UPPER), np.clip(c2, LOWER, UPPER)


def polynomial_mutation(x: np.ndarray, rng: np.random.Generator,
                        eta: float = 20.0, pm: float = None) -> np.ndarray:
    if pm is None:
        pm = 1.0 / len(x)
    y = x.copy()
    mask = rng.random(len(x)) < pm
    if not np.any(mask):
        return y
    u = rng.random(mask.sum())
    delta = np.where(u < 0.5,
                     (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0,
                     1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0)))
    y[mask] = y[mask] + delta * (UPPER - LOWER)
    return np.clip(y, LOWER, UPPER)


def tournament(rank: np.ndarray, crowd: np.ndarray, rng: np.random.Generator,
               k: int = 2) -> int:
    candidates = rng.integers(0, len(rank), size=k)
    best = candidates[0]
    for c in candidates[1:]:
        if rank[c] < rank[best]:
            best = c
        elif rank[c] == rank[best] and crowd[c] > crowd[best]:
            best = c
    return int(best)


def select_survivors(fitness: np.ndarray, n_survivors: int
                     ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Недоминируемая сортировка + crowding distance, отбор n_survivors."""
    fronts = fast_non_dominated_sort(fitness)
    selected: List[int] = []
    for front in fronts:
        if len(selected) + len(front) <= n_survivors:
            selected.extend(front.tolist())
        else:
            remaining = n_survivors - len(selected)
            cd = crowding_distance(fitness[front])
            chosen = front[np.argsort(-cd)[:remaining]]
            selected.extend(chosen.tolist())
            break

    selected = np.array(selected, dtype=int)
    rank = np.full(len(fitness), 10_000, dtype=int)
    crowd = np.zeros(len(fitness), dtype=float)
    for r, front in enumerate(fronts):
        rank[front] = r
        crowd[front] = crowding_distance(fitness[front])
    return selected, rank, fronts


def hypervolume(fitness: np.ndarray, ref: np.ndarray = REF_POINT) -> float:
    """2D гиперобъём множества точек минимизации."""
    if len(fitness) == 0:
        return 0.0
    # оставляем недоминируемые точки
    order = np.lexsort((fitness[:, 1], fitness[:, 0]))
    pts = []
    best_f2 = np.inf
    for i in order:
        f1, f2 = fitness[i]
        if f2 < best_f2 - 1e-12:
            pts.append((f1, f2))
            best_f2 = f2
    pts.sort(key=lambda p: p[0])
    hv = 0.0
    for i, (f1, f2) in enumerate(pts):
        next_f1 = pts[i + 1][0] if i + 1 < len(pts) else ref[0]
        hv += max(0.0, next_f1 - f1) * max(0.0, ref[1] - f2)
    return float(hv)


def nsga2_run(seed: int, pop_size: int = 80, generations: int = 120,
              verbose: bool = False) -> dict:
    rng = np.random.default_rng(seed)

    # Начальная популяция: случайные графики + несколько осмысленных эвристик.
    pop = rng.uniform(LOWER, UPPER, size=(pop_size, HOURS))
    heuristics = [
        np.full(HOURS, 21.0),
        np.full(HOURS, 20.0),
        np.array([18.0 if h < 7 or h >= 23 else 21.0 for h in range(HOURS)]),
        np.array([16.0 if h < 6 else (20.5 if h < 23 else 18.0) for h in range(HOURS)]),
        np.array([17.0 if h < 7 else 21.5 for h in range(HOURS)]),
    ]
    for i, h in enumerate(heuristics[:min(len(heuristics), pop_size)]):
        pop[i] = h

    fitness = evaluate(pop)
    hv_history = [hypervolume(fitness)]

    for gen in range(1, generations + 1):
        # турнирный отбор родителей по (rank, crowding)
        _, rank, _ = select_survivors(fitness, pop_size)
        _, _, fronts = select_survivors(fitness, pop_size)
        crowd = np.zeros(pop_size)
        for front in fronts:
            crowd[front] = crowding_distance(fitness[front])

        children = []
        while len(children) < pop_size:
            i1 = tournament(rank, crowd, rng)
            i2 = tournament(rank, crowd, rng)
            c1, c2 = sbx_crossover(pop[i1], pop[i2], rng)
            c1 = polynomial_mutation(c1, rng)
            c2 = polynomial_mutation(c2, rng)
            children.append(c1)
            if len(children) < pop_size:
                children.append(c2)

        children = np.array(children)
        child_fit = evaluate(children)

        combined_pop = np.vstack([pop, children])
        combined_fit = np.vstack([fitness, child_fit])
        selected, _, _ = select_survivors(combined_fit, pop_size)
        pop = combined_pop[selected]
        fitness = combined_fit[selected]
        hv_history.append(hypervolume(fitness))

        if verbose and gen % 20 == 0:
            print(f"    поколение {gen:3d}: HV = {hv_history[-1]:.2f}, "
                  f"точек = {len(fitness)}")

    return {
        "pop": pop,
        "fitness": fitness,
        "hv_history": np.array(hv_history),
        "hv": hypervolume(fitness),
    }


# ---------------------------------------------------------------------------
# Взвешенная сумма (базовый метод сравнения)
# ---------------------------------------------------------------------------
def weighted_sum_run(seed: int, weight: float, pop_size: int = 80,
                     generations: int = 120) -> dict:
    """Однокритериальный ГА: w*E/E0 + (1-w)*D/D0."""
    rng = np.random.default_rng(seed)
    pop = rng.uniform(LOWER, UPPER, size=(pop_size, HOURS))
    pop[0] = np.full(HOURS, 21.0)

    def scalar_fit(p):
        e, d, _, _ = simulate(p)
        return weight * e / 60.0 + (1.0 - weight) * d / 100.0

    fit = scalar_fit(pop)
    for _ in range(generations):
        order = np.argsort(fit)
        new_pop = [pop[order[0]].copy(), pop[order[1]].copy()]
        while len(new_pop) < pop_size:
            cand1 = rng.integers(0, pop_size, size=3)
            cand2 = rng.integers(0, pop_size, size=3)
            i1 = int(cand1[np.argmin(fit[cand1])])
            i2 = int(cand2[np.argmin(fit[cand2])])
            c1, c2 = sbx_crossover(pop[i1], pop[i2], rng)
            new_pop.append(polynomial_mutation(c1, rng))
            if len(new_pop) < pop_size:
                new_pop.append(polynomial_mutation(c2, rng))
        pop = np.array(new_pop)
        fit = scalar_fit(pop)

    best = int(np.argmin(fit))
    e, d, _, _ = simulate(pop[best:best + 1])
    return {"genome": pop[best], "energy": float(e[0]), "discomfort": float(d[0]),
            "weight": weight}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def nondominated_points(fitness: np.ndarray) -> np.ndarray:
    fronts = fast_non_dominated_sort(fitness)
    if not fronts:
        return fitness
    return fitness[fronts[0]]


def knee_point(front: np.ndarray) -> int:
    """Индекс «колена»: минимум расстояния до идеальной точки после нормировки."""
    lo = front.min(axis=0)
    hi = front.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    z = (front - lo) / span
    return int(np.argmin(np.linalg.norm(z, axis=1)))


def representative_solutions(front: np.ndarray):
    """Возвращает индексы: дешёвое, комфортное, сбалансированное."""
    cheap = int(np.argmin(front[:, 0]))
    comfortable = int(np.argmin(front[:, 1]))
    knee = knee_point(front)
    return {"cheap": cheap, "comfortable": comfortable, "balanced": knee}


def save_csv(path: str, header: List[str], rows: List[list]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def plot_block_schema(path: str):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12.5, 11.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11.8)
    ax.axis("off")

    def box(x, y, w, h, text, color="#EAF2FB", edge="#2F5D8A", fs=10):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                           linewidth=1.6, edgecolor=edge, facecolor=color)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#12263A")

    def arrow(x1, y1, x2, y2, text="", color="#2F5D8A", fs=8.5):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                            linewidth=1.6, color=color)
        ax.add_patch(a)
        if text:
            ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, text, fontsize=fs, color=color)

    ax.text(7, 11.4, "Блок-схема NSGA-II (ЛР №3, отопление помещения)",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color="#12263A")

    cx, w = 3.8, 6.0
    steps = [
        ("Старт\n(seed = 42…61)", 10.5, "#EAF2FB", "#2F5D8A"),
        ("Инициализация популяции\n24-часовых графиков температуры", 9.6, "#EAF2FB", "#2F5D8A"),
        ("Симуляция тепловой модели\n→ (энергия, дискомфорт)", 8.7, "#EAF2FB", "#2F5D8A"),
        ("Недоминируемая сортировка\nи расстояние до скопления", 7.5, "#FEF9E7", "#B7950B"),
        ("Турнирный отбор родителей\n(rank + crowding)", 6.6, "#EAF2FB", "#2F5D8A"),
        ("SBX-кроссовер + полиномиальная мутация", 5.7, "#EAF2FB", "#2F5D8A"),
        ("Объединение родителей и потомков\n(μ + λ)", 4.8, "#EAF2FB", "#2F5D8A"),
        ("Отбор N лучших по front / crowding", 3.8, "#FEF9E7", "#B7950B"),
        ("Обновление внешнего архива\nнедоминируемых решений", 2.9, "#E8F8F5", "#117A65"),
        ("Достигнуто 120 поколений?", 2.0, "#EAF2FB", "#2F5D8A"),
        ("Фронт Парето, 3 решения\nи визуализация", 0.8, "#E8F8F5", "#117A65"),
    ]
    for text, y, color, edge in steps:
        box(cx, y, w, 0.72, text, color=color, edge=edge)

    for i in range(len(steps) - 2):
        arrow(cx + w / 2, steps[i][1], cx + w / 2, steps[i + 1][1] + 0.72)

    arrow(cx + w / 2, 2.0, cx + w / 2, 1.52, text="да")

    loop_x = 12.0
    arrow(cx + w, 2.36, loop_x, 2.36, color="#B03A2E")
    arrow(loop_x, 2.36, loop_x, 7.86, color="#B03A2E")
    arrow(loop_x, 7.86, cx + w, 7.86, color="#B03A2E")
    ax.text(loop_x + 0.15, 5.0, "нет:\nследующее\nпоколение", fontsize=9,
            color="#B03A2E", ha="left", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_pareto(front: np.ndarray, weighted: np.ndarray, reps: dict, path: str):
    plt.figure(figsize=(9, 6))
    plt.scatter(weighted[:, 0], weighted[:, 1], s=38, c="#95A5A6", alpha=0.65,
                label="Взвешенная сумма (20 запусков)", zorder=2)
    order = np.argsort(front[:, 0])
    plt.plot(front[order, 0], front[order, 1], "-", color="#2F5D8A", lw=1.2, zorder=3)
    plt.scatter(front[:, 0], front[:, 1], s=55, c="#2F5D8A", label="NSGA-II: фронт Парето", zorder=4)
    labels = {"cheap": "дешёвое", "comfortable": "комфортное", "balanced": "сбалансированное"}
    colors = {"cheap": "#27AE60", "comfortable": "#C0392B", "balanced": "#8E44AD"}
    for key, idx in reps.items():
        p = front[idx]
        plt.scatter([p[0]], [p[1]], s=170, marker="*", c=colors[key],
                    edgecolor="black", linewidth=0.7, zorder=6,
                    label=f"{labels[key]} ({p[0]:.1f} кВт·ч; {p[1]:.1f} °C·ч)")
        plt.annotate(labels[key], (p[0], p[1]), textcoords="offset points",
                     xytext=(8, 8), fontsize=9, color=colors[key])
    plt.xlabel("Энергозатраты за сутки, кВт·ч")
    plt.ylabel("Дискомфорт, °C·ч (взвешенный)")
    plt.title("Фронт Парето: энергия vs комфорт")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_schedule(genomes: np.ndarray, names: List[str], path: str):
    """Температура и мощность для трёх характерных решений."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = ["#27AE60", "#C0392B", "#8E44AD"]
    e, d, temps, powers = simulate(genomes)
    for i, (name, color) in enumerate(zip(names, colors)):
        axes[0].plot(TIME, temps[i], color=color, lw=1.8,
                     label=f"{name}: {e[i]:.1f} кВт·ч, {d[i]:.1f} °C·ч")
        axes[1].step(TIME, powers[i], where="post", color=color, lw=1.3, alpha=0.85)
    axes[0].fill_between(TIME, T_LOW, T_HIGH, color="#AED6F1", alpha=0.25,
                         label="комфортная зона")
    axes[0].plot(TIME, T_OUT, color="#7F8C8D", ls="--", lw=1.2, label="улица")
    axes[0].set_ylabel("Температура, °C")
    axes[0].set_title("Суточные графики характерных решений")
    axes[0].legend(fontsize=8.5, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("Мощность отопления, кВт")
    axes[1].set_xlabel("Час суток")
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 24)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_hv(hv_matrix: np.ndarray, path: str):
    mean = hv_matrix.mean(axis=0)
    std = hv_matrix.std(axis=0)
    gens = np.arange(len(mean))
    plt.figure(figsize=(9, 5))
    plt.plot(gens, mean, color="#2F5D8A", lw=2, label="NSGA-II: средний HV")
    plt.fill_between(gens, mean - std, mean + std, color="#2F5D8A", alpha=0.15)
    plt.xlabel("Поколение")
    plt.ylabel("Гиперобъём (больше — лучше)")
    plt.title("Сходимость NSGA-II по гиперобъёму (20 запусков)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_dashboard(front, weighted, reps, genomes, names, hv_matrix, stats, path):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(17, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)
    fig.suptitle("ЛР №3 · NSGA-II для режима отопления · фронт Парето и компромиссы",
                 fontsize=16, fontweight="bold", color="#12263A", y=0.98)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(weighted[:, 0], weighted[:, 1], s=32, c="#95A5A6", alpha=0.6,
                label="взвешенная сумма")
    order = np.argsort(front[:, 0])
    ax1.plot(front[order, 0], front[order, 1], color="#2F5D8A", lw=1.2)
    ax1.scatter(front[:, 0], front[:, 1], s=42, c="#2F5D8A", label="NSGA-II")
    colors = {"cheap": "#27AE60", "comfortable": "#C0392B", "balanced": "#8E44AD"}
    labels = {"cheap": "дешёвое", "comfortable": "комфортное", "balanced": "сбалансированное"}
    for key, idx in reps.items():
        p = front[idx]
        ax1.scatter([p[0]], [p[1]], s=160, marker="*", c=colors[key],
                    edgecolor="black", linewidth=0.6, zorder=5)
        ax1.annotate(labels[key], (p[0], p[1]), textcoords="offset points",
                     xytext=(7, 7), fontsize=9, color=colors[key])
    ax1.set_xlabel("Энергозатраты, кВт·ч")
    ax1.set_ylabel("Дискомфорт, °C·ч")
    ax1.set_title("Фронт Парето и характерные решения")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2 = fig.add_subplot(gs[0, 2])
    hv_mean = hv_matrix.mean(axis=0)
    hv_std = hv_matrix.std(axis=0)
    gens = np.arange(len(hv_mean))
    ax2.plot(gens, hv_mean, color="#2F5D8A", lw=2)
    ax2.fill_between(gens, hv_mean - hv_std, hv_mean + hv_std, color="#2F5D8A", alpha=0.15)
    ax2.set_xlabel("Поколение")
    ax2.set_ylabel("Гиперобъём")
    ax2.set_title("Сходимость (mean ± std)")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :2])
    e, d, temps, powers = simulate(genomes)
    for i, (name, color) in enumerate(zip(names, ["#27AE60", "#C0392B", "#8E44AD"])):
        ax3.plot(TIME, temps[i], color=color, lw=1.7,
                 label=f"{name}: {e[i]:.1f} кВт·ч / {d[i]:.1f} °C·ч")
    ax3.fill_between(TIME, T_LOW, T_HIGH, color="#AED6F1", alpha=0.22, label="комфорт")
    ax3.plot(TIME, T_OUT, color="#7F8C8D", ls="--", lw=1.1, label="улица")
    ax3.set_xlabel("Час суток")
    ax3.set_ylabel("Температура, °C")
    ax3.set_title("Суточные температурные графики")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8, ncol=2)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    lines = [
        "Сводка по 20 запускам NSGA-II",
        "",
        f"HV: {stats['hv_mean']:.1f} ± {stats['hv_std']:.1f}",
        f"Точек на фронте: {stats['points_mean']:.1f}",
        f"Лучшая энергия: {stats['min_energy']:.2f} кВт·ч",
        f"Лучший комфорт: {stats['min_discomfort']:.2f} °C·ч",
        f"Взвешенная сумма: {stats['weighted_points']} решений",
        "",
        "Вывод: NSGA-II даёт набор",
        "компромиссов, а не одно решение.",
    ]
    ax4.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#EAF2FB",
                       edgecolor="#2F5D8A"))
    ax4.set_title("Метрики", pad=10)

    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------
def main():
    os.makedirs("lab3_out", exist_ok=True)
    print("ЛР №3: NSGA-II для режима отопления (вариант 4, трек A)")
    print(f"Тепловая модель: C={C_ROOM} кВт·ч/К, UA={UA} кВт/К, Pmax={P_MAX} кВт")

    all_runs = []
    hv_matrix = []
    print(f"\nЗапуск NSGA-II: {N_RUNS} независимых запусков...")
    for r in range(N_RUNS):
        res = nsga2_run(BASE_SEED + r, verbose=False)
        all_runs.append(res)
        hv_matrix.append(res["hv_history"])
        front = nondominated_points(res["fitness"])
        print(f"  запуск {r + 1:02d}: HV={res['hv']:8.1f}, точек={len(front):2d}, "
              f"minE={front[:, 0].min():5.2f}, minD={front[:, 1].min():6.2f}")

    hv_matrix = np.array(hv_matrix)
    # лучший запуск по HV для детальной визуализации
    best_run_idx = int(np.argmax([r["hv"] for r in all_runs]))
    best = all_runs[best_run_idx]
    front_idx = fast_non_dominated_sort(best["fitness"])[0]
    front = best["fitness"][front_idx]

    reps = representative_solutions(front)
    rep_names = ["дешёвое", "комфортное", "сбалансированное"]
    rep_keys = ["cheap", "comfortable", "balanced"]
    rep_indices = [reps[k] for k in rep_keys]
    rep_genomes = best["pop"][front_idx[rep_indices]]

    # Взвешенная сумма: 20 запусков со случайными весами
    print("\nЗапуск взвешенной суммы (20 запусков)...")
    weighted_rows = []
    weighted_points = []
    rng = np.random.default_rng(BASE_SEED + 500)
    for r in range(N_RUNS):
        w = float(rng.uniform(0.0, 1.0))
        res = weighted_sum_run(BASE_SEED + 1000 + r, w)
        weighted_rows.append([r, w, res["energy"], res["discomfort"]])
        weighted_points.append([res["energy"], res["discomfort"]])
    weighted_points = np.array(weighted_points)
    weighted_front = nondominated_points(weighted_points)

    # CSV: фронт лучшего запуска
    save_csv("lab3_out/lab3_pareto.csv",
             ["energy_kwh", "discomfort_degree_hours"],
             [[float(a), float(b)] for a, b in front[np.argsort(front[:, 0])]])

    # CSV: метрики запусков
    run_rows = []
    for r, res in enumerate(all_runs):
        f = nondominated_points(res["fitness"])
        run_rows.append([r, res["hv"], len(f), float(f[:, 0].min()), float(f[:, 1].min())])
    save_csv("lab3_out/lab3_runs.csv",
             ["run", "hypervolume", "pareto_points", "min_energy", "min_discomfort"],
             run_rows)

    # CSV: сходимость HV
    hv_rows = []
    for g in range(hv_matrix.shape[1]):
        hv_rows.append([g, float(hv_matrix[:, g].mean()), float(hv_matrix[:, g].std())])
    save_csv("lab3_out/lab3_hv.csv", ["generation", "hv_mean", "hv_std"], hv_rows)

    # CSV: взвешенная сумма
    save_csv("lab3_out/lab3_weighted.csv",
             ["run", "weight", "energy_kwh", "discomfort_degree_hours"],
             weighted_rows)

    # CSV: характерные решения (геном + метрики)
    rep_rows = []
    for key, idx, name in zip(rep_keys, [reps[k] for k in rep_keys], rep_names):
        p = front[idx]
        rep_rows.append([name, float(p[0]), float(p[1])]
                        + [float(v) for v in rep_genomes[rep_keys.index(key)]])
    save_csv("lab3_out/lab3_representatives.csv",
             ["solution", "energy_kwh", "discomfort_degree_hours"]
             + [f"target_h{h}" for h in range(HOURS)], rep_rows)

    # CSV: суточный график трёх решений
    e3, d3, temps3, powers3 = simulate(rep_genomes)
    sched_rows = []
    for k in range(STEPS):
        row = [float(TIME[k]), float(T_OUT[k])]
        for i in range(3):
            row += [float(temps3[i, k]), float(powers3[i, k])]
        sched_rows.append(row)
    save_csv("lab3_out/lab3_schedule.csv",
             ["hour", "t_out"] + sum([[f"{n}_temp", f"{n}_power"] for n in rep_names], []),
             sched_rows)

    # Сводка
    stats = {
        "variant": 4,
        "track": "A",
        "runs": N_RUNS,
        "best_run": best_run_idx,
        "hv_mean": float(hv_matrix[:, -1].mean()),
        "hv_std": float(hv_matrix[:, -1].std()),
        "points_mean": float(np.mean([len(nondominated_points(r["fitness"])) for r in all_runs])),
        "min_energy": float(front[:, 0].min()),
        "min_discomfort": float(front[:, 1].min()),
        "weighted_points": int(len(weighted_front)),
        "reference_point": REF_POINT.tolist(),
    }
    with open("lab3_out/lab3_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== Сводка ===")
    print(f"Лучший запуск: {best_run_idx}, HV = {best['hv']:.1f}")
    print(f"Точек на фронте (среднее): {stats['points_mean']:.1f}")
    print(f"Минимальная энергия: {stats['min_energy']:.2f} кВт·ч")
    print(f"Минимальный дискомфорт: {stats['min_discomfort']:.2f} °C·ч")
    print(f"Гиперобъём: {stats['hv_mean']:.1f} ± {stats['hv_std']:.1f}")

    print("\nПостроение графиков...")
    plot_block_schema("lab3_out/lab3_block_schema.png")
    plot_pareto(front, weighted_points, reps, "lab3_out/lab3_pareto.png")
    plot_hv(hv_matrix, "lab3_out/lab3_hv.png")
    plot_schedule(rep_genomes, rep_names, "lab3_out/lab3_schedule.png")
    plot_dashboard(front, weighted_points, reps, rep_genomes, rep_names,
                   hv_matrix, stats, "lab3_out/lab3_dashboard.png")
    print("Готово. Файлы в lab3_out/.")


if __name__ == "__main__":
    main()
