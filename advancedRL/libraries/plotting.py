import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_behavior(results, world, n_trials, window_size=5):
    """
    Plot trial-by-trial behavior and feature learning for a single game.

    Parameters
    ----------
    results : dict
        Output of Agent.simulate_choice_task (single game).
    world : World
        Task environment (used for n_feats and target feature).
    n_trials : int
        Number of trials in the simulated game.
    window_size : int, optional
        Window size for smoothed accuracy curve.
    """

    # Set plotting style with white background
    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Feature weights
    feature_weights = results["feature_weights"]
    target_feat_idx = results["target"] - 1  # Convert to 0-indexed
    non_target_mask = np.arange(world.n_feats) != target_feat_idx

    # 1. Accuracy over time (proportion of trials where target feature was chosen)
    accuracy_smooth = (
        pd.Series(results["correct"])
        .rolling(window=window_size, min_periods=1)
        .mean()
    )

    axes[0, 0].plot(
        range(1, n_trials + 1),
        accuracy_smooth,
        linewidth=2,
        label="Smoothed accuracy",
        color="#333333",
    )
    axes[0, 0].axhline(
        y=1 / 3, color="r", linestyle="--", alpha=0.5, label="Chance (33%)"
    )
    axes[0, 0].set_xlabel("Trial", fontsize=14)
    axes[0, 0].set_ylabel("Proportion Correct", fontsize=14)
    axes[0, 0].set_title(
        "Learning Curve: Accuracy Over Time", fontsize=14, fontweight="bold"
    )
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].set_xticks(range(1, n_trials + 1))  # One tick per trial
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_facecolor("white")
    axes[0, 0].tick_params(labelsize=12)

    # 2. Feature weights over time (target vs non-target average)
    axes[0, 1].plot(
        range(1, n_trials + 1),
        feature_weights[:, target_feat_idx],
        linewidth=2,
        label=f"Target feature ({results['target']})",
        color="#333333",
        linestyle="-",
    )
    non_target_avg = np.mean(feature_weights[:, non_target_mask], axis=1)
    axes[0, 1].plot(
        range(1, n_trials + 1),
        non_target_avg,
        linewidth=2,
        label="Non-target (average)",
        color="#333333",
        alpha=0.7,
        linestyle="--",
    )
    axes[0, 1].set_xlabel("Trial", fontsize=14)
    axes[0, 1].set_ylabel("Feature Weight", fontsize=14)
    axes[0, 1].set_title(
        "Feature Weights: Target vs Non-Target", fontsize=14, fontweight="bold"
    )
    axes[0, 1].set_xticks(range(1, n_trials + 1))  # One tick per trial
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_facecolor("white")
    axes[0, 1].tick_params(labelsize=12)

    # 3. All feature weights learning over time
    markers = ["o", "s", "^", "v", "D", "p"]
    feature_colors = {}

    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    tab10_list = [tab10_colors[i] for i in range(10)]

    for feat_idx in range(world.n_feats):
        feat_num = feat_idx + 1
        marker = markers[feat_idx % len(markers)]
        if feat_idx == target_feat_idx:
            feature_colors[feat_num] = "green"
            axes[1, 0].plot(
                range(1, n_trials + 1),
                feature_weights[:, feat_idx],
                linewidth=2.5,
                label=f"Feature {feat_num} (TARGET)",
                color="#333333",
                linestyle="-",
            )
        else:
            feature_colors[feat_num] = tab10_list[feat_idx % len(tab10_list)]
            axes[1, 0].plot(
                range(1, n_trials + 1),
                feature_weights[:, feat_idx],
                linewidth=1.5,
                label=f"Feature {feat_num}",
                color="#333333",
                alpha=0.8,
                linestyle="--",
                marker=marker,
                markersize=8,
                markevery=max(1, n_trials // 8),
            )

    axes[1, 0].set_xlabel("Trial", fontsize=14)
    axes[1, 0].set_ylabel("Feature Weight", fontsize=14)
    axes[1, 0].set_title(
        "All Feature Weights Over Time", fontsize=14, fontweight="bold"
    )
    axes[1, 0].set_xticks(range(1, n_trials + 1))  # One tick per trial
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_facecolor("white")
    axes[1, 0].tick_params(labelsize=12)

    # 4. Trial-by-trial choices: features chosen on each trial
    choices = results["choices"]
    outcomes = results["outcomes"]

    vlag_colors = plt.cm.RdYlBu_r
    reward_color = vlag_colors(0.2)  # Muted blue
    no_reward_color = vlag_colors(0.8)  # Muted red

    for t in range(n_trials):
        chosen_features = choices[t]
        outcome = outcomes[t]
        color = reward_color if outcome == 1 else no_reward_color

        for feat in chosen_features:
            axes[1, 1].scatter(
                t + 1,
                feat,
                s=100,
                marker="s",
                color=color,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )

    axes[1, 1].set_xlabel("Trial", fontsize=14)
    axes[1, 1].set_ylabel("Feature", fontsize=14)
    axes[1, 1].set_title(
        "Trial-by-Trial Choices: Features Selected", fontsize=14, fontweight="bold"
    )
    axes[1, 1].tick_params(labelsize=12)
    axes[1, 1].set_ylim([0.5, world.n_feats + 0.5])
    axes[1, 1].set_yticks(range(1, world.n_feats + 1))
    axes[1, 1].set_xlim([-1, n_trials + 0.5])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_facecolor("white")
    axes[1, 1].set_xticks(range(1, n_trials + 1))

    # Y-axis tick labels and markers matching feature plot
    ytick_labels = [f"F{i}" for i in range(1, world.n_feats + 1)]
    axes[1, 1].set_yticklabels(ytick_labels)
    markers_local = ["o", "s", "^", "v", "D", "p"]
    for i, label in enumerate(axes[1, 1].get_yticklabels()):
        feat_num = i + 1
        feat_idx = feat_num - 1
        label.set_color("black")
        label.set_fontweight("bold" if feat_num == results["target"] else "normal")
        if feat_num != results["target"]:
            marker = markers_local[feat_idx % len(markers_local)]
            axes[1, 1].plot(
                -0.3,
                feat_num,
                marker=marker,
                color="#333333",
                markersize=8,
                markeredgecolor="#333333",
                markeredgewidth=0.5,
                clip_on=False,
                zorder=10,
            )

    legend_elements = [
        Patch(
            facecolor=reward_color,
            alpha=0.7,
            edgecolor="black",
            label="Reward",
        ),
        Patch(
            facecolor=no_reward_color,
            alpha=0.7,
            edgecolor="black",
            label="No reward",
        ),
    ]
    axes[1, 1].legend(handles=legend_elements, loc="upper right")

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.show()


def plot_attention_conditions(
    avg_accuracy_congruent,
    sem_accuracy_congruent,
    avg_accuracy_neutral,
    sem_accuracy_neutral,
    avg_accuracy_incongruent,
    sem_accuracy_incongruent,
    n_trials,
    n_agents,
    phi_congruent,
    phi_neutral,
    phi_incongruent,
    target_dim,
):
    """
    Plot average learning curves for congruent, neutral, and incongruent attention conditions.

    Parameters
    ----------
    avg_accuracy_... : np.ndarray
        Average accuracy over time for each condition.
    sem_accuracy_... : np.ndarray
        Standard error of the mean over time for each condition.
    n_trials : int
        Number of trials per game.
    n_agents : int
        Number of agents per condition.
    phi_congruent, phi_neutral, phi_incongruent : float
        Attention parameters for each condition.
    target_dim : int
        0-indexed dimension index containing the target feature (for labeling).
    """

    vlag_colors = plt.cm.RdYlBu_r
    congruent_color = vlag_colors(0.2)  # muted blue
    neutral_color = "#666666"  # gray
    incongruent_color = vlag_colors(0.8)  # muted red

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor("white")

    # Congruent
    congruent_label = (
        f"Congruent (φ={phi_congruent}, more attn to target dim {target_dim + 1})"
    )
    ax.plot(
        range(1, n_trials + 1),
        avg_accuracy_congruent,
        linewidth=2.5,
        color=congruent_color,
        label=congruent_label,
        zorder=3,
    )
    ax.fill_between(
        range(1, n_trials + 1),
        avg_accuracy_congruent - sem_accuracy_congruent,
        avg_accuracy_congruent + sem_accuracy_congruent,
        alpha=0.2,
        color=congruent_color,
        zorder=2,
    )

    # Neutral
    neutral_label = f"Neutral (φ={phi_neutral}, equal attention)"
    ax.plot(
        range(1, n_trials + 1),
        avg_accuracy_neutral,
        linewidth=2.5,
        color=neutral_color,
        label=neutral_label,
        zorder=3,
        linestyle="--",
    )
    ax.fill_between(
        range(1, n_trials + 1),
        avg_accuracy_neutral - sem_accuracy_neutral,
        avg_accuracy_neutral + sem_accuracy_neutral,
        alpha=0.2,
        color=neutral_color,
        zorder=2,
    )

    # Incongruent
    incongruent_label = (
        f"Incongruent (φ={phi_incongruent}, less attn to target dim {target_dim + 1})"
    )
    ax.plot(
        range(1, n_trials + 1),
        avg_accuracy_incongruent,
        linewidth=2.5,
        color=incongruent_color,
        label=incongruent_label,
        zorder=3,
    )
    ax.fill_between(
        range(1, n_trials + 1),
        avg_accuracy_incongruent - sem_accuracy_incongruent,
        avg_accuracy_incongruent + sem_accuracy_incongruent,
        alpha=0.2,
        color=incongruent_color,
        zorder=2,
    )

    # Chance line
    ax.axhline(
        y=1 / 3,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Chance (33%)",
        zorder=1,
    )

    ax.set_xlabel("Trial", fontsize=14)
    ax.set_ylabel("Proportion Correct", fontsize=14)
    ax.set_title(
        f"Average Learning Curves: Attention Conditions\n(n = {n_agents} agents per condition)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim([0, 1])
    ax.set_xticks(range(1, n_trials + 1))
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.show()

