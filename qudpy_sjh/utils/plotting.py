"""Plotting helpers that draw on matplotlib axes and return handles."""

from __future__ import annotations

import numpy as np

from qudpy_sjh.utils.core.results import DynamicsResult

def _line_styles(count: int) -> list:
    """
    Return a list of visually distinguishable matplotlib line styles.

    The first four are standard named styles. The later ones use custom dash
    patterns. If count exceeds the list length, styles are repeated cyclically.
    """
    base = [
        "-",                    # solid
        "--",                   # dashed
        "-.",                   # dash-dot
        ":",                    # dotted

        (0, (5, 1)),            # dense dashed
        (0, (3, 1, 1, 1)),      # dense dash-dot
        (0, (1, 1)),            # dense dotted
        (0, (7, 2)),            # long dashed

        (0, (7, 2, 1, 2)),      # long dash-dot
        (0, (9, 2, 2, 2)),      # long dash short dash
        (0, (3, 2, 1, 2, 1, 2)),
        (0, (5, 2, 5, 2, 1, 2)),

        (0, (10, 3)),
        (0, (10, 3, 2, 3)),
        (0, (2, 2)),
        (0, (2, 2, 8, 2)),

        (0, (1, 3)),
        (0, (4, 3)),
        (0, (6, 3, 1, 3)),
        (0, (8, 3, 1, 3, 1, 3)),
    ]

    if count <= len(base):
        return base[:count]

    return [base[i % len(base)] for i in range(count)]

def _plasma_colors(count: int, *, start: float = 0.18, end: float = 0.88):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("plasma")
    if count <= 1:
        return [cmap((start + end) * 0.5)]
    return list(cmap(np.linspace(start, end, count)))


def _new_axes(ax=None, *, figsize=(8, 4)):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    fig, created_ax = plt.subplots(figsize=figsize)
    return fig, created_ax


def _new_axes_array(axes=None, *, nrows: int = 2, ncols: int = 1, figsize=(8, 6), sharex=True):
    import matplotlib.pyplot as plt

    if axes is not None:
        axes_array = np.asarray(axes, dtype=object)
        return axes_array.flat[0].figure, axes_array
    fig, created_axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex)
    return fig, np.asarray(created_axes, dtype=object)


def _mode_title(result: DynamicsResult) -> str:
    return {
        "lab_exact": "Lab frame",
        "rotating_view": "Rotating view",
        "rwa": "RWA",
    }.get(result.mode, result.mode)


def _times_and_label(result: DynamicsResult) -> tuple[np.ndarray, str]:
    if result.times_fs is not None:
        return result.times_fs, "Time (fs)"
    return result.times, "Time"


def _upper_triangular_pairs(dimension: int, max_pairs: int | None = None) -> list[tuple[int, int]]:
    pairs = [(i, j) for i in range(dimension) for j in range(i + 1, dimension)]
    if max_pairs is not None:
        return pairs[:max_pairs]
    return pairs


def _normalize_axis_choice(value) -> str:
    if value in (0, "0", "left", "Left", "LEFT", "l", "L"):
        return "left"
    if value in (1, "1", "right", "Right", "RIGHT", "r", "R"):
        return "right"
    raise ValueError(f"Unsupported axis choice: {value!r}. Use 'left'/'right' or 0/1.")


def plot_field(
    field,
    times,
    ax=None,
    label: str | None = None,
    *,
    ylabel: str = "field (code unit)",
    plot_times=None,
):
    fig, ax = _new_axes(ax)
    code_times = np.asarray(times, dtype=float)
    shown_times = code_times if plot_times is None else np.asarray(plot_times, dtype=float)
    values = np.asarray(field(code_times), dtype=float)
    ax.plot(shown_times, values, label=label or getattr(field, "name", "field"), color=_plasma_colors(1)[0])
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_drive(
    result: DynamicsResult,
    ax=None,
    times: np.ndarray | None = None,
    label: str | None = None,
    *,
    max_points: int = 2000,
    display_code_unit: bool = False,
):
    fig, ax = _new_axes(ax)
    if result.drive is None:
        ax.text(0.5, 0.5, "derived from lab drive", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("input")
        ax.grid(True, alpha=0.3)
        return fig, ax

    sample_times_code = result.times if times is None else np.asarray(times, dtype=float)
    shown_times = _times_and_label(result)[0] if times is None else np.asarray(times, dtype=float)
    if sample_times_code.size > max_points:
        stride = int(np.ceil(sample_times_code.size / max_points))
        sample_times_code = sample_times_code[::stride]
        shown_times = shown_times[::stride]

    if result.mode == "lab_exact" and not display_code_unit and result.times_fs is not None:
        values = result.field_MV_per_cm_values(sample_times_code, times_fs=shown_times)
        if values is not None:
            ax.plot(shown_times, values, label=label or "physical field", color=_plasma_colors(1)[0])
            ax.set_ylabel("E(t) (MV/cm)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            return fig, ax

    if result.mode == "rwa" and not display_code_unit:
        values = result.drive_fs_inv_values(sample_times_code)
        if values is not None:
            ax.plot(shown_times, values, label=label or "Omega(t) (fs^-1)", color=_plasma_colors(1)[0])
            ax.set_ylabel("Omega(t) (fs^-1)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            return fig, ax

    values = result.drive_code_values(sample_times_code)
    if result.mode == "rwa":
        ylabel = "Omega(t) (code unit)"
        default_label = "RWA drive (code unit)"
    else:
        ylabel = "carrier (code unit)"
        default_label = "normalized carrier"
    ax.plot(shown_times, values, label=label or default_label, color=_plasma_colors(1)[0])
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_populations(result: DynamicsResult, ax=None, populations=None, *, title: str | None = None):
    fig, ax = _new_axes(ax)
    times, time_label = _times_and_label(result)
    density = result.density_array()
    population_indices = list(range(result.dimension())) if populations is None else list(populations)
    colors = _plasma_colors(len(population_indices))

    for color, index in zip(colors, population_indices):
        ax.plot(times, density[:, index, index].real, label=fr"$\rho_{{{index}{index}}}$", color=color)
    ax.set_xlabel(time_label)
    ax.set_ylabel("Population")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_population_preview(
    result: DynamicsResult,
    ax=None,
    *,
    populations=None,
    title: str | None = None,
    population_axis_map: dict[int, str | int] | None = None,
    split_population_axes_threshold: float = 0.1,
):
    fig, ax = _new_axes(ax)
    times, time_label = _times_and_label(result)
    density = result.density_array()

    population_indices = list(range(result.dimension())) if populations is None else list(populations)
    curves = {index: density[:, index, index].real for index in population_indices}

    total_span = float(sum(np.nanmax(values) - np.nanmin(values) for values in curves.values()))
    use_single_axis = total_span > split_population_axes_threshold or len(population_indices) <= 1

    if population_axis_map is None:
        normalized_axis_map = {
            index: ("left" if index == 0 else "right")
            for index in population_indices
        }
    else:
        normalized_axis_map = {}
        for index in population_indices:
            default_side = "left" if index == 0 else "right"
            side = population_axis_map.get(index, default_side)
            normalized_axis_map[index] = _normalize_axis_choice(side)

    if use_single_axis:
        styles = _line_styles(len(population_indices))
        line_handles = []
        line_labels = []

        for linestyle, index in zip(styles, population_indices):
            line, = ax.plot(
                times,
                curves[index],
                label=fr"$\rho_{{{index}{index}}}$",
                color="black",
                linestyle=linestyle,
                linewidth=2.0,
            )
            line_handles.append(line)
            line_labels.append(line.get_label())

        ax.set_ylabel("Population")
        ax.legend(line_handles, line_labels, loc="best")

    else:
        from itertools import cycle

        ax_right = ax.twinx()

        left_handles = []
        right_handles = []

        base_styles = _line_styles(4)
        left_style_iter = cycle(base_styles)
        right_style_iter = cycle(base_styles)

        for index in population_indices:
            axis_side = normalized_axis_map[index]

            if axis_side == "left":
                target_ax = ax
                color = "black"
                linestyle = next(left_style_iter)
            else:
                target_ax = ax_right
                color = "red"
                linestyle = next(right_style_iter)

            label = fr"$\rho_{{{index}{index}}}$ ({axis_side})"

            line, = target_ax.plot(
                times,
                curves[index],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
            )

            if axis_side == "left":
                left_handles.append(line)
            else:
                right_handles.append(line)

        ax.set_ylabel("Population (left)")
        ax_right.set_ylabel("Population (right)")

        ax.yaxis.label.set_color("black")
        ax.tick_params(axis="y", colors="black")
        ax.spines["left"].set_color("black")

        ax_right.yaxis.label.set_color("red")
        ax_right.tick_params(axis="y", colors="red")
        ax_right.spines["right"].set_color("red")

        handles = left_handles + right_handles
        labels = [handle.get_label() for handle in handles]
        ax.legend(handles, labels, loc="best")

    ax.set_xlabel(time_label)
    if title is not None:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    return fig, ax

def plot_coherences(
    result: DynamicsResult,
    ax=None,
    coherences=None,
    *,
    title: str | None = None,
    max_pairs: int | None = 6,
):
    fig, ax = _new_axes(ax)
    times, time_label = _times_and_label(result)
    if coherences is None:
        coherences = _upper_triangular_pairs(result.dimension(), max_pairs=max_pairs)
    colors = _plasma_colors(max(2, 2 * len(coherences)))

    for pair_index, (i, j) in enumerate(coherences):
        values = result.matrix_element(i, j)
        label = fr"$\rho_{{{i}{j}}}$"
        color_re = colors[2 * pair_index]
        color_im = colors[2 * pair_index + 1]
        ax.plot(times, values.real, label=f"Re({label})", color=color_re)
        ax.plot(times, values.imag, linestyle="--", label=f"Im({label})", color=color_im)
    ax.set_xlabel(time_label)
    ax.set_ylabel("Coherence")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_coherence_abs(
    result: DynamicsResult,
    ax=None,
    coherences=None,
    *,
    title: str | None = None,
    max_pairs: int | None = 6,
):
    fig, ax = _new_axes(ax)
    times, time_label = _times_and_label(result)
    if coherences is None:
        coherences = _upper_triangular_pairs(result.dimension(), max_pairs=max_pairs)
    colors = _plasma_colors(len(coherences))
    for color, (i, j) in zip(colors, coherences):
        values = result.matrix_element(i, j)
        ax.plot(times, np.abs(values), label=fr"$|\rho_{{{i}{j}}}|$", color=color)
    ax.set_xlabel(time_label)
    ax.set_ylabel("|Coherence|")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_coherence_phases(
    result: DynamicsResult,
    ax=None,
    coherences=None,
    *,
    title: str | None = None,
    max_pairs: int | None = 6,
    mask_threshold: float = 1e-8,
):
    fig, ax = _new_axes(ax)
    times, time_label = _times_and_label(result)
    if coherences is None:
        coherences = _upper_triangular_pairs(result.dimension(), max_pairs=max_pairs)
    colors = _plasma_colors(len(coherences))
    for color, (i, j) in zip(colors, coherences):
        values = result.matrix_element(i, j)
        phase = np.unwrap(np.angle(values)).astype(float)
        phase[np.abs(values) < mask_threshold] = np.nan
        ax.plot(times, phase, label=fr"$phase(\rho_{{{i}{j}}})$", color=color)
    ax.set_xlabel(time_label)
    ax.set_ylabel("Phase (rad)")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_density_components(
    result: DynamicsResult,
    axes=None,
    *,
    include_drive: bool = False,
    title: str | None = None,
    display_code_unit: bool = False,
):
    nrows = 3 if include_drive else 2
    fig, axes_array = _new_axes_array(axes, nrows=nrows, ncols=1, figsize=(4.9, 1.68 * nrows), sharex=True)
    axes_flat = axes_array.reshape(-1)
    row = 0
    if include_drive:
        plot_drive(result, ax=axes_flat[row], display_code_unit=display_code_unit)
        axes_flat[row].set_title(title if title is not None else _mode_title(result))
        axes_flat[row].set_xlabel("")
        row += 1
    else:
        axes_flat[row].set_title(title if title is not None else _mode_title(result))
    plot_populations(result, ax=axes_flat[row])
    axes_flat[row].set_xlabel("")
    row += 1
    default_coherences = [(0, 1)] if result.dimension() == 2 else None
    plot_coherences(result, ax=axes_flat[row], coherences=default_coherences)
    return fig, axes_array


def plot_multilevel_components(
    result: DynamicsResult,
    axes=None,
    *,
    populations: list[int] | tuple[int, ...] | None = None,
    coherences: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    title: str | None = None,
    max_pairs: int | None = 6,
):
    if coherences is None and result.dimension() >= 2:
        coherences = _upper_triangular_pairs(result.dimension(), max_pairs=max_pairs)
    n_rows = 1 if not coherences else 2
    fig, axes_array = _new_axes_array(axes, nrows=n_rows, ncols=1, figsize=(7.0, 2.8 + 2.1 * (n_rows - 1)), sharex=True)
    axes_flat = axes_array.reshape(-1)
    plot_populations(result, ax=axes_flat[0], populations=populations, title=title if title is not None else "Multi-level")
    if coherences:
        plot_coherences(result, ax=axes_flat[1], coherences=coherences, max_pairs=max_pairs)
        axes_flat[0].set_xlabel("")
    return fig, axes_array


def build_preview_figure(
    result: DynamicsResult,
    *,
    coherences=None,
    display_code_unit: bool = False,
    max_pairs: int | None = 6,
    population_axis_map: dict[int, str | int] | None = None,
    split_population_axes_threshold: float = 0.1,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(8.6, 13.0), sharex=True)
    selected_coherences = coherences
    if selected_coherences is None:
        selected_coherences = _upper_triangular_pairs(result.dimension(), max_pairs=max_pairs)

    plot_drive(result, ax=axes[0], display_code_unit=display_code_unit)
    axes[0].set_title(_mode_title(result))
    axes[0].set_xlabel("")

    plot_population_preview(
        result,
        ax=axes[1],
        population_axis_map=population_axis_map,
        split_population_axes_threshold=split_population_axes_threshold,
    )
    axes[1].set_xlabel("")

    plot_coherence_abs(result, ax=axes[2], coherences=selected_coherences, max_pairs=max_pairs)
    axes[2].set_xlabel("")

    plot_coherence_phases(result, ax=axes[3], coherences=selected_coherences, max_pairs=max_pairs)

    fig.tight_layout()
    return fig, axes


def build_component_figures(
    result: DynamicsResult,
    *,
    coherences=None,
    max_pairs: int | None = 6,
):
    import matplotlib.pyplot as plt

    times, time_label = _times_and_label(result)
    selected_coherences = coherences
    if selected_coherences is None:
        selected_coherences = _upper_triangular_pairs(result.dimension(), max_pairs=max_pairs)

    figures: list[tuple[str, object]] = []

    for (i, j) in selected_coherences:
        values = result.matrix_element(i, j)

        fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True)

        axes[0].plot(
            times,
            values.real,
            color="black",
            linestyle="-",
            linewidth=2.0,
            label=fr"$\mathrm{{Re}}(\rho_{{{i}{j}}})$",
        )
        axes[0].set_title(fr"$\rho_{{{i}{j}}}$ components")
        axes[0].set_ylabel("Real part")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(
            times,
            values.imag,
            color="#E6B800",
            linestyle="-",
            linewidth=2.0,
            label=fr"$\mathrm{{Im}}(\rho_{{{i}{j}}})$",
        )
        axes[1].set_xlabel(time_label)
        axes[1].set_ylabel("Imag part")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        figures.append((f"rho_{i}{j}", fig))

    return figures
def normalize_for_shape(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if np.iscomplexobj(y):
        y = np.real(y)
    y = y.astype(float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return y
    scale = np.nanmax(np.abs(y[finite]))
    if scale == 0.0:
        return y
    return y / scale


def sorted_xy_for_plot(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    return x[order], y[order]


def plot_normalized_curve(ax, x, y, *, label: str, **plot_kwargs) -> np.ndarray:
    x, y = sorted_xy_for_plot(x, y)
    if x.size == 0:
        return np.array([], dtype=float)
    y_norm = normalize_for_shape(y)
    ax.plot(x, y_norm, label=label, **plot_kwargs)
    return y_norm


def real_if_close_or_abs_for_plot(values: np.ndarray) -> tuple[np.ndarray, str]:
    values = np.asarray(values)
    real_values = np.real_if_close(values, tol=1000)
    if np.iscomplexobj(real_values):
        return np.abs(values), "|g(t)|"
    return np.asarray(real_values, dtype=float), "g(t)"


def add_top_omega_axis(ax, x_pos=0.13, y_pos=0.92):
    from qudpy_sjh.utils.core.normalization import ParaNormalizer

    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda energy_eV: energy_eV * ParaNormalizer.EV_TO_FS_INV,
            lambda omega_fs_inv: omega_fs_inv / ParaNormalizer.EV_TO_FS_INV,
        ),
    )
    secax.set_xlabel("")
    secax.tick_params(axis="x", pad=1)
    ax.text(
        x_pos,
        y_pos,
        "Ang. Freq. (fs$^{-1}$)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        clip_on=False,
    )
    return secax


def set_energy_axis(ax, e_min: float, e_max: float, *, n_ticks: int = 3):
    ax.set_xlim(e_min, e_max)
    ax.set_xticks(np.linspace(e_min, e_max, n_ticks))


def set_axis_ylim_from_curves(ax, curves: list[np.ndarray], *, min_headroom: float = 0.12):
    finite_values = []
    for curve in curves:
        curve = np.asarray(curve, dtype=float)
        finite = curve[np.isfinite(curve)]
        if finite.size > 0:
            finite_values.append(finite)
    if not finite_values:
        return
    all_values = np.concatenate(finite_values)
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    if y_min == y_max:
        if y_max == 0.0:
            y_min, y_max = -1.0, 1.0
        else:
            pad = 0.1 * abs(y_max)
            y_min -= pad
            y_max += pad
    span = y_max - y_min
    pad = max(min_headroom * span, 0.05)
    ax.set_ylim(y_min - pad, y_max + pad)


__all__ = [
    "plot_field",
    "plot_drive",
    "plot_populations",
    "plot_population_preview",
    "plot_coherences",
    "plot_coherence_abs",
    "plot_coherence_phases",
    "plot_density_components",
    "plot_multilevel_components",
    "build_preview_figure",
    "build_component_figures",
    "add_top_omega_axis",
    "normalize_for_shape",
    "plot_normalized_curve",
    "real_if_close_or_abs_for_plot",
    "set_axis_ylim_from_curves",
    "set_energy_axis",
]