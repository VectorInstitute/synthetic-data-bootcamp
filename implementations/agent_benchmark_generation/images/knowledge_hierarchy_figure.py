"""Draw the SynBench knowledge-hierarchy figure for the mock retail domain.

matplotlib is not a project dependency, so run this standalone:

    uv run --with matplotlib python images/knowledge_hierarchy_figure.py
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl


mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


OUT = Path(__file__).resolve().parent / "knowledge_hierarchy.png"

GEN_HEADER, GEN_FACE = "#7b2d8e", "#f6ecf9"
SIM_HEADER, SIM_FACE = "#0f766e", "#e7f5f3"
AGENT_HEADER, AGENT_FACE = "#1f4e79", "#eaf1f8"
CODE_HEADER, CODE_FACE = "#4a5568", "#f1f2f4"
ART_HEADER, ART_FACE = "#8a6d1f", "#fdf6e3"
SEES, HIDDEN, INK = "#166534", "#b91c1c", "#1a1a1a"

HEADER_H = 5.2
PAD_TOP = 2.6
PAD_BOT = 2.2
LINE_H = 3.4
FS = 8.3

Point = tuple[float, float]


def panel_height(n_lines: int, *, subtitle: bool) -> float:
    """Return the height of a panel holding ``n_lines`` body lines."""
    return HEADER_H + PAD_TOP + (LINE_H if subtitle else 0) + n_lines * LINE_H + PAD_BOT


def box(
    ax: Any,
    span: Point,
    ytop: float,
    title: str,
    lines: Sequence[str],
    *,
    face: str,
    header: str,
    subtitle: str | None = None,
) -> float:
    """Draw a titled panel anchored at its top edge and return its bottom edge.

    Body lines are marked up by their first character: ``+`` renders a green
    "this role sees it" bullet, ``-`` a red "withheld" bullet, ``!`` bold plain
    text, anything else muted plain text.
    """
    x0, x1 = span
    ybot = ytop - panel_height(len(lines), subtitle=subtitle is not None)
    ax.add_patch(
        FancyBboxPatch(
            (x0, ybot),
            x1 - x0,
            ytop - ybot,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=1.3,
            edgecolor=header,
            facecolor=face,
            zorder=2,
        ),
    )
    ax.add_patch(
        FancyBboxPatch(
            (x0, ytop - HEADER_H),
            x1 - x0,
            HEADER_H,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=0,
            facecolor=header,
            zorder=3,
        ),
    )
    ax.text(
        (x0 + x1) / 2,
        ytop - HEADER_H / 2,
        title,
        ha="center",
        va="center",
        fontsize=FS + 1.4,
        fontweight="bold",
        color="white",
        zorder=4,
    )
    y = ytop - HEADER_H - PAD_TOP - LINE_H / 2
    if subtitle:
        ax.text(
            x0 + 2.0,
            y,
            subtitle,
            ha="left",
            va="center",
            fontsize=FS - 0.6,
            style="italic",
            color="#4a4a4a",
            zorder=4,
        )
        y -= LINE_H
    for line in lines:
        if line.startswith("+"):
            color, text, weight = SEES, "\u2713 " + line[1:].strip(), "normal"
        elif line.startswith("-"):
            color, text, weight = HIDDEN, "\u2717 " + line[1:].strip(), "normal"
        elif line.startswith("!"):
            color, text, weight = INK, line[1:].strip(), "bold"
        else:
            color, text, weight = "#333333", line, "normal"
        ax.text(x0 + 2.0, y, text, ha="left", va="center", fontsize=FS, color=color, fontweight=weight, zorder=4)
        y -= LINE_H
    return ybot


def arrow(
    ax: Any,
    p0: Point,
    p1: Point,
    label: str | None = None,
    *,
    dashed: bool = False,
    color: str = "#333333",
    lp: float = 0.5,
    off: Point = (0, 2.2),
    fs: float = 7.9,
    rad: float = 0.0,
) -> None:
    """Connect two points, optionally labelling the arrow at fraction ``lp``."""
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            linestyle=(0, (5, 3)) if dashed else "-",
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1,
            shrinkB=1,
            zorder=5,
        ),
    )
    if label:
        ax.text(
            p0[0] + (p1[0] - p0[0]) * lp + off[0],
            p0[1] + (p1[1] - p0[1]) * lp + off[1],
            label,
            ha="center",
            va="center",
            fontsize=fs,
            color=color,
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.92},
        )


def band(ax: Any, span: Point, extent: Point, label: str, color: str) -> None:
    """Draw a dashed frame around one phase of the pipeline, labelled on its left."""
    x0, x1 = span
    y0, y1 = extent
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0,rounding_size=1.5",
            linewidth=1.0,
            linestyle=(0, (6, 4)),
            edgecolor=color,
            facecolor="none",
            zorder=1,
        ),
    )
    ax.text(
        x0 - 3.0,
        (y0 + y1) / 2,
        label,
        rotation=90,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=color,
    )


def main() -> None:
    """Render the figure to ``knowledge_hierarchy.png``."""
    fig, ax = plt.subplots(figsize=(17.5, 11.9))
    ax.set_xlim(0, 178)
    ax.set_ylim(-23, 131)
    ax.axis("off")

    ax.text(
        6,
        127,
        "SynBench knowledge hierarchy \u2014 mock retail",
        ha="left",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        6,
        122.5,
        "Every role calls the same model API; only the prompt differs. Each box lists what that LLM is given "
        "(\u2713) and what is withheld from it (\u2717).",
        ha="left",
        va="center",
        fontsize=10,
        color="#444444",
    )

    # ----------------------------------------------------------- generation
    db_bot = box(
        ax,
        (10, 50),
        116,
        "db.json",
        ["users: user_id, name, email", "orders: order_id, user_id, status,", "     items, shipping_address"],
        face=CODE_FACE,
        header=CODE_HEADER,
    )
    sampler_bot = box(
        ax,
        (10, 50),
        db_bot - 5,
        "ConstraintSampler (code)",
        [
            "samples a task_type from task_types.yaml,",
            "one order row + its joined user,",
            "one personality_style",
            "! \u2192 entity_context (real IDs)",
        ],
        face=CODE_FACE,
        header=CODE_HEADER,
    )
    gen_bot = box(
        ax,
        (58, 112),
        112,
        "Generator LLM",
        [
            "+ policy.md, tool specs (READ / WRITE)",
            "+ allow_write rule for the sampled task_type",
            "+ seed tasks as few-shot oracles",
            "+ entity_context: order_id, user_id, status,",
            "     address, and the joined user name / email",
            "! most privileged \u2014 nothing is withheld",
            "! Because it already knows the IDs, it can write the",
            "! shortest oracle; the agent has to earn those IDs.",
        ],
        face=GEN_FACE,
        header=GEN_HEADER,
        subtitle="writes the ground truth",
    )
    task_bot = box(
        ax,
        (120, 174),
        112,
        "Task JSON  (verified draft \u2192 tasks.json)",
        [
            "! description + task_type",
            "     the evaluation goal \u2014 shown to no LLM",
            "! user_scenario",
            "     user_name, personality_style, instructions,",
            "     initial_message  \u2192 user simulator only",
            "! evaluation_criteria",
            "     oracle actions, communicate_info, reward_basis",
            "     \u2192 scorer only",
        ],
        face=ART_FACE,
        header=ART_HEADER,
    )

    band(ax, (7, 177), (min(sampler_bot, gen_bot, task_bot) - 4, 119), "GENERATION TIME  (notebook 2)", GEN_HEADER)

    arrow(ax, (30, db_bot), (30, db_bot - 5), label="sample", off=(7.5, 0))
    sampler_mid = (db_bot - 5 + sampler_bot) / 2
    arrow(ax, (50, sampler_mid), (58, sampler_mid), label="entity_context", off=(0, 2.6), fs=7.2)
    arrow(ax, (112, 100), (120, 100), label="writes draft\n+ oracle", off=(0, 4.4), fs=7.2)
    ax.text(
        116,
        89,
        "verify_draft:\nrules \u2192 task type\n\u2192 replay",
        ha="center",
        va="center",
        fontsize=6.9,
        color="#555555",
        style="italic",
    )

    # ----------------------------------------------------------- evaluation
    top = 56
    sim_bot = box(
        ax,
        (10, 58),
        top,
        "User simulator LLM",
        [
            "+ user_name + personality_style description",
            "+ instructions (its own private brief)",
            "+ initial_message (turn 0, sent verbatim)",
            "+ live transcript",
            "- policy.md, the tools, the database",
            "- task.description, oracle actions",
            "! Reveals IDs when asked, not all at once;",
            "! ends the episode with [[DONE]].",
        ],
        face=SIM_FACE,
        header=SIM_HEADER,
        subtitle="plays the customer",
    )
    agent_bot = box(
        ax,
        (68, 116),
        top,
        "Agent under test",
        [
            "+ policy.md (agent_system_prompt)",
            "+ tool specs and tool results",
            "+ customer messages only",
            "- instructions, task.description",
            "- oracle actions, communicate_info",
            "- raw database contents",
            "! Must elicit the full name, look up the",
            "! user_id, then mutate \u2014 nothing is handed over.",
        ],
        face=AGENT_FACE,
        header=AGENT_HEADER,
        subtitle="least privileged \u2014 the system being measured",
    )
    env_bot = box(
        ax,
        (126, 162),
        top,
        "Environment",
        [
            "per-episode copy of db.json",
            "(the on-disk file is never mutated)",
            "! READ: find_user_id, list_orders,",
            "!           get_order",
            "! WRITE: cancel_order, update_shipping",
            "returns tool results, never raw tables",
        ],
        face=CODE_FACE,
        header=CODE_HEADER,
        subtitle="tools read and mutate the DB copy",
    )

    row2 = min(sim_bot, agent_bot, env_bot) - 6
    legend_bot = box(
        ax,
        (10, 58),
        row2,
        "Reading this figure",
        [
            "! \u2713  in that role's prompt",
            "! \u2717  withheld from that role",
            "solid arrows: live dialogue and tool calls",
            "dashed arrows: data routed by code, unseen",
        ],
        face="#ffffff",
        header="#666666",
    )
    pipe_bot = box(
        ax,
        (68, 116),
        row2,
        "Notebook 4: planner / executor / critic",
        [
            "planner sees policy + live conversation,",
            "critic sees plan, tool trace, draft reply",
            "same blind spots as the single agent",
            "! only the executor may call tools",
        ],
        face=AGENT_FACE,
        header=AGENT_HEADER,
    )
    score_bot = box(
        ax,
        (126, 174),
        row2,
        "score_trajectory (code)",
        [
            "replays agent actions on a fresh DB",
            "DB: final hash == oracle hash",
            "COMMUNICATE: required substrings",
            "! reward = DB \u00d7 COMMUNICATE",
        ],
        face=CODE_FACE,
        header=CODE_HEADER,
    )

    band(
        ax,
        (7, 177),
        (min(legend_bot, pipe_bot, score_bot) - 4, 61),
        "EVALUATION TIME  (notebooks 3 & 4)",
        AGENT_HEADER,
    )

    # task JSON fields routed into the evaluation band
    arrow(
        ax,
        (128, task_bot),
        (44, top),
        label="user_scenario",
        dashed=True,
        color=SIM_HEADER,
        lp=0.55,
        off=(0, 5.0),
        rad=0.06,
    )
    arrow(
        ax,
        (168, task_bot),
        (168, row2),
        label="evaluation_criteria",
        dashed=True,
        color=ART_HEADER,
        lp=0.42,
        off=(0, 0),
    )
    ax.text(
        93,
        64.5,
        "task.description  \u2717  never reaches the simulator, the agent, the planner or the critic",
        ha="center",
        va="center",
        fontsize=8.6,
        color=HIDDEN,
        fontweight="bold",
    )

    # dialogue, tools and scoring
    mid_hi, mid_lo = 40, 28
    arrow(ax, (58, mid_hi), (68, mid_hi), label="customer turns", off=(0, 2.6))
    arrow(ax, (68, mid_lo), (58, mid_lo), label="agent replies", off=(0, -2.8))
    ax.text(63, 19, "natural\nlanguage\nonly", ha="center", va="center", fontsize=7.3, color="#666666", style="italic")
    arrow(ax, (116, mid_hi), (126, mid_hi), label="tool calls", off=(0, 2.6))
    arrow(ax, (126, mid_lo), (116, mid_lo), label="results", off=(0, -2.8))

    arrow(ax, (92, agent_bot), (92, row2), label="swap-in variant", color=AGENT_HEADER, off=(-12.0, 0), fs=7.5)
    arrow(ax, (144, env_bot), (144, row2), label="final DB state", dashed=True, off=(-11, 0), fs=7.5)
    arrow(
        ax,
        (116, agent_bot + 4),
        (126, row2 + 3),
        label="agent actions\n+ messages",
        dashed=True,
        off=(-6.5, -4.6),
        fs=7.5,
        rad=-0.15,
    )

    fig.savefig(OUT, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
