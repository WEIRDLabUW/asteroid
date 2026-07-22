import argparse
import csv
import os
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from environments.rollout_policy import TransformerCNNPolicy
from models import DecisionTransformerCnn
from environments.procgen_env import make_maze_envs
from experiments.train_procgen_bc_ppo import RecurrentBCPPOPolicy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ours_policy(run_dir, checkpoint, temp):
    run_dir = Path(run_dir)
    with open(run_dir / "model_args.pkl", "rb") as f:
        model_args = pickle.load(f)
    ckpt = Path(checkpoint) if checkpoint else run_dir / "final_model.pth"
    model = DecisionTransformerCnn(model_args).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return TransformerCNNPolicy(
        model=model,
        context_horizon=model_args["horizon"],
        temp=temp,
    )


def load_bc_policy(run_dir, checkpoint, obs_shape, action_dim):
    run_dir = Path(run_dir)
    args_path = run_dir / "args.json"
    hidden_dim = 128
    obs_dim = 128
    if args_path.exists():
        import json
        with open(args_path) as f:
            cfg = json.load(f)
        hidden_dim = int(cfg.get("hidden_dim", hidden_dim))
        obs_dim = int(cfg.get("obs_dim", obs_dim))

    ckpt = Path(checkpoint) if checkpoint else run_dir / "final_model.pt"
    model = RecurrentBCPPOPolicy(
        obs_shape=obs_shape,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        obs_dim=obs_dim,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def make_eval_env(args):
    n_eval = args.n_eval_envs if args.env_split == "eval" else args.n_train_envs
    train_env, eval_env = make_maze_envs(
        n_train=args.n_train_envs,
        n_eval=n_eval,
        train_start=args.train_start_level,
        train_levels=args.train_num_levels,
        eval_start=args.eval_start_level,
        eval_levels=args.eval_num_levels,
        visibility=args.visibility,
        local_window_obs=True,
    )
    return train_env if args.env_split == "train" else eval_env


def reset_to_states(env, states):
    env.env.set_state(states)
    env._solve_all()
    _, raw_obs, _ = env.env.observe()
    obs = env._build_grid_obs()
    infos = env._build_infos(raw_rgb=raw_obs["rgb"])
    return obs, infos


def init_maps(infos):
    maps = []
    layouts = []
    for info in infos:
        nav = info["nav"].astype(bool)
        maps.append(np.zeros_like(nav, dtype=np.float32))
        layouts.append({
            "nav": nav.copy(),
            "goal_pos": info.get("goal_pos"),
            "world_dim": int(info.get("world_dim", nav.shape[0])),
        })
    return maps, layouts


def rollout_ours(policy, env, horizon, initial_states=None):
    if initial_states is None:
        obs, infos = env.reset()
    else:
        obs, infos = reset_to_states(env, initial_states)
    maps, layouts = init_maps(infos)
    done = np.zeros(env.n, dtype=bool)
    successes = np.zeros(env.n, dtype=bool)
    returns = np.zeros(env.n, dtype=np.float32)
    lengths = np.zeros(env.n, dtype=np.int32)
    policy.reset(np.ones(env.n, dtype=bool))

    fallback_actions = np.zeros(env.n, dtype=np.int64)
    for _ in range(horizon):
        for i, info in enumerate(infos):
            if done[i] or "agent_pos" not in info:
                continue
            x, y = info["agent_pos"]
            maps[i][y, x] += 1.0

        actions = policy.get_action(obs)
        actions = np.where(done, fallback_actions, actions)
        next_obs, rewards, env_dones, next_infos = env.step(actions)
        active = ~done
        returns[active] += rewards[active]
        lengths[active] += 1
        dones = env_dones | done
        for i in range(env.n):
            if active[i] and env_dones[i]:
                successes[i] = bool(next_infos[i].get("success", False))
        update_rewards = np.where(active, rewards, 0.0)
        policy.update_context(obs, actions, update_rewards, dones)
        policy.reset(dones)
        done = done | env_dones
        obs, infos = next_obs, next_infos
        if done.all():
            break
    stats = {"successes": successes, "returns": returns, "lengths": lengths}
    return maps, layouts, stats


@torch.no_grad()
def rollout_bc(model, env, horizon, initial_states=None):
    if initial_states is None:
        obs, infos = env.reset()
    else:
        obs, infos = reset_to_states(env, initial_states)
    maps, layouts = init_maps(infos)
    done = np.zeros(env.n, dtype=bool)
    successes = np.zeros(env.n, dtype=bool)
    returns = np.zeros(env.n, dtype=np.float32)
    lengths = np.zeros(env.n, dtype=np.int32)
    hidden = None
    prev_actions = np.zeros(env.n, dtype=np.int64)
    prev_rewards = np.zeros(env.n, dtype=np.float32)
    prev_dones = np.ones(env.n, dtype=np.float32)

    fallback_actions = np.zeros(env.n, dtype=np.int64)
    for _ in range(horizon):
        for i, info in enumerate(infos):
            if done[i] or "agent_pos" not in info:
                continue
            x, y = info["agent_pos"]
            maps[i][y, x] += 1.0

        actions, hidden = model.act(
            obs,
            prev_actions,
            prev_rewards,
            prev_dones,
            hidden=hidden,
            deterministic=True,
        )
        actions = np.where(done, fallback_actions, actions)
        next_obs, rewards, env_dones, next_infos = env.step(actions)
        active = ~done
        returns[active] += rewards[active]
        lengths[active] += 1
        dones = env_dones | done
        for i in range(env.n):
            if active[i] and env_dones[i]:
                successes[i] = bool(next_infos[i].get("success", False))
        prev_actions = actions.astype(np.int64)
        prev_rewards = np.where(active, rewards, 0.0).astype(np.float32)
        prev_dones = dones.astype(np.float32)
        if hidden is not None and dones.any():
            done_idx = torch.as_tensor(dones, dtype=torch.bool, device=hidden.device)
            hidden[:, done_idx, :] = 0.0
        done = done | env_dones
        obs, infos = next_obs, next_infos
        if done.all():
            break
    stats = {"successes": successes, "returns": returns, "lengths": lengths}
    return maps, layouts, stats


def draw_maze_heatmap(ax, visits, layout, title, vmax, success=None, episode_return=None):
    nav = layout["nav"]
    data = np.flipud(visits)
    wall_mask = np.flipud(~nav)

    sns.heatmap(
        data,
        mask=wall_mask,
        ax=ax,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
        linewidths=0.25,
        linecolor="#e6e6e6",
    )
    ax.set_facecolor("#171717")

    goal = layout.get("goal_pos")
    if goal is not None:
        gx, gy = goal
        row = nav.shape[0] - 1 - gy
        ax.scatter(
            [gx + 0.5],
            [row + 0.5],
            marker="*",
            s=70,
            c="#d62728",
            edgecolors="white",
            linewidths=0.5,
            zorder=10,
        )

    if success is not None and episode_return is not None:
        status = "success" if success else "failure"
        title = f"{title}\n{status}, return={episode_return:.0f}"
    ax.set_title(title, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")


def success_rate(stats, indices):
    if len(indices) == 0:
        return 0.0
    return float(np.mean([stats["successes"][idx] for idx in indices]))


def write_metrics_csv(path, bc_stats, ours_stats, left_label, selected_indices):
    rows = []
    for plot_idx, maze_idx in enumerate(selected_indices):
        for label, stats in [(left_label, bc_stats), ("Ours", ours_stats)]:
            rows.append({
                "plot_idx": plot_idx,
                "maze_idx": int(maze_idx),
                "policy": label,
                "success": int(bool(stats["successes"][maze_idx])),
                "return": float(stats["returns"][maze_idx]),
                "length": int(stats["lengths"][maze_idx]),
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmaps(
    bc_maps,
    ours_maps,
    layouts,
    output_path,
    n_mazes,
    left_label="BC policy",
    bc_stats=None,
    ours_stats=None,
    save_individual=True,
    require_both_success=True,
    max_count=50.0,
):
    sns.set_theme(
        context="paper",
        style="white",
        font="DejaVu Sans",
        rc={
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        },
    )

    max_available = min(len(layouts), len(bc_maps), len(ours_maps))
    if require_both_success and bc_stats is not None and ours_stats is not None:
        selected_indices = [
            idx for idx in range(max_available)
            if bool(bc_stats["successes"][idx]) and bool(ours_stats["successes"][idx])
        ]
        if len(selected_indices) == 0:
            raise ValueError("No mazes where both BC and Ours succeeded; cannot make requested heatmap.")
        if len(selected_indices) < n_mazes:
            print(
                f"Only {len(selected_indices)} mazes had both policies succeed; "
                f"plotting those instead of requested {n_mazes}."
            )
        selected_indices = selected_indices[:n_mazes]
    else:
        selected_indices = list(range(min(n_mazes, max_available)))
    n_mazes = len(selected_indices)

    fig, axes = plt.subplots(
        n_mazes,
        2,
        figsize=(4.8, max(2.4, 2.15 * n_mazes)),
        squeeze=False,
    )

    vmax = float(max_count)

    bc_sr = success_rate(bc_stats, selected_indices) if bc_stats is not None else 0.0
    ours_sr = success_rate(ours_stats, selected_indices) if ours_stats is not None else 0.0
    for plot_idx, maze_idx in enumerate(selected_indices):
        draw_maze_heatmap(
            axes[plot_idx, 0],
            bc_maps[maze_idx],
            layouts[maze_idx],
            left_label if plot_idx == 0 else "",
            vmax,
            success=bool(bc_stats["successes"][maze_idx]) if bc_stats is not None else None,
            episode_return=float(bc_stats["returns"][maze_idx]) if bc_stats is not None else None,
        )
        draw_maze_heatmap(
            axes[plot_idx, 1],
            ours_maps[maze_idx],
            layouts[maze_idx],
            "Ours" if plot_idx == 0 else "",
            vmax,
            success=bool(ours_stats["successes"][maze_idx]) if ours_stats is not None else None,
            episode_return=float(ours_stats["returns"][maze_idx]) if ours_stats is not None else None,
        )
        axes[plot_idx, 0].set_ylabel(f"Maze {maze_idx + 1}", rotation=90, labelpad=8)

    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist())
    cbar.set_label("Visitation count", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    handles = [
        mpl.lines.Line2D([0], [0], marker="*", color="none", markerfacecolor="#d62728",
                         markeredgecolor="white", markersize=9, label="Goal"),
        mpl.patches.Patch(facecolor="#171717", edgecolor="#171717", label="Wall"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        framealpha=1.0,
        fontsize=8,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if bc_stats is not None and ours_stats is not None:
        write_metrics_csv(
            output_path.with_suffix(".csv"),
            bc_stats,
            ours_stats,
            left_label,
            selected_indices,
        )
        fig.suptitle(
            f"{left_label} success={bc_sr:.2f}   |   Ours success={ours_sr:.2f}",
            fontsize=9,
            y=0.995,
        )
    fig.subplots_adjust(left=0.08, right=0.88, top=0.94, bottom=0.09, wspace=0.12, hspace=0.18)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")
    if bc_stats is not None and ours_stats is not None:
        print(f"Saved {output_path.with_suffix('.csv')}")

    if save_individual:
        individual_dir = output_path.parent / f"{output_path.stem}_individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        for plot_idx, maze_idx in enumerate(selected_indices):
            fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.6), squeeze=False)
            draw_maze_heatmap(
                axes[0, 0],
                bc_maps[maze_idx],
                layouts[maze_idx],
                left_label,
                vmax,
                success=bool(bc_stats["successes"][maze_idx]) if bc_stats is not None else None,
                episode_return=float(bc_stats["returns"][maze_idx]) if bc_stats is not None else None,
            )
            draw_maze_heatmap(
                axes[0, 1],
                ours_maps[maze_idx],
                layouts[maze_idx],
                "Ours",
                vmax,
                success=bool(ours_stats["successes"][maze_idx]) if ours_stats is not None else None,
                episode_return=float(ours_stats["returns"][maze_idx]) if ours_stats is not None else None,
            )
            fig.suptitle(f"Maze {maze_idx + 1}", fontsize=9, y=0.98)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.04, pad=0.03)
            cbar.set_label("Visitation count", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            fig.subplots_adjust(left=0.04, right=0.88, top=0.84, bottom=0.04, wspace=0.12)
            individual_path = individual_dir / f"maze_{maze_idx + 1:02d}.png"
            fig.savefig(individual_path, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        print(f"Saved individual maze plots to {individual_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot BC-policy vs Ours state-visitation heatmaps on procgen mazes."
    )
    parser.add_argument("--ours-run-dir", required=True)
    parser.add_argument("--ours-checkpoint", default=None)
    parser.add_argument("--bc-run-dir", required=True)
    parser.add_argument("--bc-checkpoint", default=None)
    parser.add_argument("--bc-policy-type", choices=["bc_ppo", "context"], default="bc_ppo")
    parser.add_argument("--output", default="procgen_plots/visitation_heatmaps_bc_vs_ours.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-split", choices=["train", "eval"], default="eval")
    parser.add_argument("--n-train-envs", type=int, default=1)
    parser.add_argument("--n-eval-envs", type=int, default=6)
    parser.add_argument("--n-plot-mazes", type=int, default=6)
    parser.add_argument("--visibility", type=int, default=3)
    parser.add_argument("--train-start-level", type=int, default=0)
    parser.add_argument("--train-num-levels", type=int, default=1000)
    parser.add_argument("--eval-start-level", type=int, default=1000)
    parser.add_argument("--eval-num-levels", type=int, default=1000)
    parser.add_argument("--eval-horizon", type=int, default=800)
    parser.add_argument("--ours-temp", type=float, default=0.1)
    parser.add_argument("--bc-temp", type=float, default=0.1)
    parser.add_argument("--no-individual", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--max-count", type=float, default=50.0)
    return parser


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_env = make_eval_env(args)
    base_env.reset()
    initial_states = base_env.env.get_state()

    bc_env = make_eval_env(args)
    if args.bc_policy_type == "context":
        bc_policy = load_ours_policy(args.bc_run_dir, args.bc_checkpoint, args.bc_temp)
        bc_maps, layouts, bc_stats = rollout_ours(
            bc_policy, bc_env, args.eval_horizon, initial_states=initial_states
        )
        left_label = "BC"
    else:
        bc_policy = load_bc_policy(
            args.bc_run_dir,
            args.bc_checkpoint,
            obs_shape=bc_env.observation_space.shape,
            action_dim=bc_env.action_space.n,
        )
        bc_maps, layouts, bc_stats = rollout_bc(
            bc_policy, bc_env, args.eval_horizon, initial_states=initial_states
        )
        left_label = "BC+PPO"

    ours_env = make_eval_env(args)
    ours_policy = load_ours_policy(args.ours_run_dir, args.ours_checkpoint, args.ours_temp)
    ours_maps, ours_layouts, ours_stats = rollout_ours(
        ours_policy, ours_env, args.eval_horizon, initial_states=initial_states
    )
    for idx, (left_layout, right_layout) in enumerate(zip(layouts, ours_layouts)):
        if (
            not np.array_equal(left_layout["nav"], right_layout["nav"])
            or left_layout["goal_pos"] != right_layout["goal_pos"]
        ):
            raise RuntimeError(f"BC and Ours layouts differ at env {idx}; refusing to plot.")

    plot_heatmaps(
        bc_maps=bc_maps,
        ours_maps=ours_maps,
        layouts=layouts,
        output_path=args.output,
        n_mazes=args.n_plot_mazes,
        left_label=left_label,
        bc_stats=bc_stats,
        ours_stats=ours_stats,
        save_individual=not args.no_individual,
        require_both_success=not args.allow_failures,
        max_count=args.max_count,
    )


if __name__ == "__main__":
    main()
