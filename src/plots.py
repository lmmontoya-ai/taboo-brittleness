import matplotlib.pyplot as plt


def plot_token_probability(
    all_probs, token_id, tokenizer, input_words, figsize=(22, 11), start_idx=0
):
    """Plot the probability of a specific token across all positions and layers."""
    # Get the probability of the specific token across all layers and positions
    token_probs = all_probs[:, start_idx:, token_id]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Set default font and increase font size
    plt.rcParams.update({"font.size": 30})

    # Create heatmap    
    im = ax.imshow(
        token_probs,
        cmap="viridis",
        aspect="auto",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=32)

    # Set labels
    ax.set_ylabel("Layers", fontsize=36)

    # Set y-ticks (layers) - only show every 4th tick
    all_yticks = list(range(token_probs.shape[0]))
    ax.set_yticks(all_yticks[::4])
    ax.tick_params(axis="y", labelsize=32)

    # Set x-ticks (tokens)
    if len(input_words) > 0:
        ax.set_xticks(list(range(len(input_words[start_idx:]))))
        ax.set_xticklabels(
            input_words[start_idx:], rotation=75, ha="right", fontsize=30
        )

    # Adjust layout
    plt.tight_layout()

    return fig