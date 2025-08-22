# plots.py
import matplotlib.pyplot as plt


def plot_token_probability(
  all_probs,
  token_id,
  tokenizer,
  input_words,
  figsize=(22, 11),
  start_idx=0,
  font_size=30,
  title_font_size=36,
  tick_font_size=32,
  colormap="viridis",
):
  """Plot probability of a specific token across all layers/positions."""
  token_probs = all_probs[:, start_idx:, token_id]

  fig, ax = plt.subplots(figsize=figsize)
  plt.rcParams.update({"font.size": font_size})

  im = ax.imshow(
    token_probs,
    cmap=colormap,
    aspect="auto",
    vmin=0,
    vmax=1,
    interpolation="nearest",
  )

  cbar = fig.colorbar(im, ax=ax)
  cbar.ax.tick_params(labelsize=tick_font_size)

  ax.set_ylabel("Layers", fontsize=title_font_size)

  all_yticks = list(range(token_probs.shape[0]))
  if len(all_yticks) > 0:
    ax.set_yticks(all_yticks[::4])
  ax.tick_params(axis="y", labelsize=tick_font_size)

  if len(input_words) > 0:
    ax.set_xticks(list(range(len(input_words[start_idx:]))))
    ax.set_xticklabels(
      input_words[start_idx:], rotation=75, ha="right", fontsize=font_size
    )

  plt.tight_layout()
  return fig