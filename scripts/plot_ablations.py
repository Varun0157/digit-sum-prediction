"""Generate bar plots for ablation study results."""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.size'] = 11

OUTPUT_DIR = 'static/main'

# Data from ablation results (accuracy in %, MAE as raw values)
results = {
    'baseline': {'val': 93.07, 'test': 92.63, 'val_digit': 96.87, 'val_mae': 0.311, 'test_mae': 0.315},
    'k3': {'val': 91.67, 'test': 91.97, 'val_digit': 96.50, 'val_mae': 0.390, 'test_mae': 0.345},
    'k5': {'val': 93.40, 'test': 93.07, 'val_digit': 96.92, 'val_mae': 0.316, 'test_mae': 0.313},
    'sum0.5': {'val': 93.33, 'test': 92.43, 'val_digit': 96.93, 'val_mae': 0.289, 'test_mae': 0.303},
    'sum1.0': {'val': 92.90, 'test': 92.33, 'val_digit': 96.81, 'val_mae': 0.303, 'test_mae': 0.311},
    'w1.25': {'val': 92.13, 'test': 91.03, 'val_digit': 96.63, 'val_mae': 0.356, 'test_mae': 0.411},
    'w1.50': {'val': 93.33, 'test': 92.93, 'val_digit': 96.93, 'val_mae': 0.302, 'test_mae': 0.320},
    'aug': {'val': 94.63, 'test': 93.63, 'val_digit': 97.25, 'val_mae': 0.251, 'test_mae': 0.283},
    'spatial': {'val': 90.87, 'test': 90.50, 'val_digit': 96.26, 'val_mae': 0.402, 'test_mae': 0.413},
}

def plot_width_comparison():
    """Width multiplier ablation."""
    configs = ['w1.0 (baseline)', 'w1.25', 'w1.50']
    val_accs = [results['baseline']['val'], results['w1.25']['val'], results['w1.50']['val']]
    test_accs = [results['baseline']['test'], results['w1.25']['test'], results['w1.50']['test']]
    val_maes = [results['baseline']['val_mae'], results['w1.25']['val_mae'], results['w1.50']['val_mae']]
    test_maes = [results['baseline']['test_mae'], results['w1.25']['test_mae'], results['w1.50']['test_mae']]
    params = ['1.22M', '2.06M', '2.96M']

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, val_accs, width, label='Val Acc', color='#5B9BD5')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Acc', color='#ED7D31')

    ax.set_ylabel('Sum Accuracy (%)')
    ax.set_title('Width Multiplier Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}\n({p})' for c, p in zip(configs, params)])
    ax.set_ylim(88, 96)

    # Add MAE on secondary axis (distinct colors for visibility)
    ax2 = ax.twinx()
    ax2.plot(x, val_maes, 'o--', color='#9467BD', label='Val MAE', markersize=8, linewidth=2)
    ax2.plot(x, test_maes, 's--', color='#2CA02C', label='Test MAE', markersize=8, linewidth=2)
    ax2.set_ylabel('MAE')
    ax2.set_ylim(0.25, 0.45)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=8)

    # Add value labels for bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/width_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUTPUT_DIR}/width_comparison.png')

def plot_kernel_comparison():
    """Kernel size ablation."""
    configs = ['k3', 'k5', 'k7 (baseline)']
    val_accs = [results['k3']['val'], results['k5']['val'], results['baseline']['val']]
    test_accs = [results['k3']['test'], results['k5']['test'], results['baseline']['test']]
    val_maes = [results['k3']['val_mae'], results['k5']['val_mae'], results['baseline']['val_mae']]
    test_maes = [results['k3']['test_mae'], results['k5']['test_mae'], results['baseline']['test_mae']]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, val_accs, width, label='Val Acc', color='#5B9BD5')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Acc', color='#ED7D31')

    ax.set_ylabel('Sum Accuracy (%)')
    ax.set_title('Initial Kernel Size Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(88, 96)

    # Add MAE on secondary axis (distinct colors for visibility)
    ax2 = ax.twinx()
    ax2.plot(x, val_maes, 'o--', color='#9467BD', label='Val MAE', markersize=8, linewidth=2)
    ax2.plot(x, test_maes, 's--', color='#2CA02C', label='Test MAE', markersize=8, linewidth=2)
    ax2.set_ylabel('MAE')
    ax2.set_ylim(0.25, 0.45)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=8)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/kernel_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUTPUT_DIR}/kernel_comparison.png')

def plot_sumloss_comparison():
    """Sum loss weight ablation."""
    configs = ['sum=0 (baseline)', 'sum=0.5', 'sum=1.0']
    val_accs = [results['baseline']['val'], results['sum0.5']['val'], results['sum1.0']['val']]
    test_accs = [results['baseline']['test'], results['sum0.5']['test'], results['sum1.0']['test']]
    val_maes = [results['baseline']['val_mae'], results['sum0.5']['val_mae'], results['sum1.0']['val_mae']]
    test_maes = [results['baseline']['test_mae'], results['sum0.5']['test_mae'], results['sum1.0']['test_mae']]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, val_accs, width, label='Val Acc', color='#5B9BD5')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Acc', color='#ED7D31')

    ax.set_ylabel('Sum Accuracy (%)')
    ax.set_title('Sum Loss Regularization Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(88, 96)

    # Add MAE on secondary axis (distinct colors for visibility)
    ax2 = ax.twinx()
    ax2.plot(x, val_maes, 'o--', color='#9467BD', label='Val MAE', markersize=8, linewidth=2)
    ax2.plot(x, test_maes, 's--', color='#2CA02C', label='Test MAE', markersize=8, linewidth=2)
    ax2.set_ylabel('MAE')
    ax2.set_ylim(0.25, 0.35)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=8)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sumloss_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUTPUT_DIR}/sumloss_comparison.png')

def plot_augmentation_comparison():
    """Augmentation comparison - dual axis format like others."""
    configs = ['Baseline', '+ Augmentation']
    val_accs = [results['baseline']['val'], results['aug']['val']]
    test_accs = [results['baseline']['test'], results['aug']['test']]
    val_maes = [results['baseline']['val_mae'], results['aug']['val_mae']]
    test_maes = [results['baseline']['test_mae'], results['aug']['test_mae']]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, val_accs, width, label='Val Acc', color='#5B9BD5')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Acc', color='#ED7D31')

    ax.set_ylabel('Sum Accuracy (%)')
    ax.set_title('Effect of Data Augmentation')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(90, 96)

    # Add MAE on secondary axis (distinct colors for visibility)
    ax2 = ax.twinx()
    ax2.plot(x, val_maes, 'o--', color='#9467BD', label='Val MAE', markersize=8, linewidth=2)
    ax2.plot(x, test_maes, 's--', color='#2CA02C', label='Test MAE', markersize=8, linewidth=2)
    ax2.set_ylabel('MAE')
    ax2.set_ylim(0.20, 0.40)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=8)

    # Add value labels for bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/augmentation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUTPUT_DIR}/augmentation_comparison.png')

def plot_spatial_comparison():
    """Spatial attention vs baseline."""
    configs = ['Baseline\n(Global Avg Pool)', 'Spatial Attention\n(Per-Head)']
    val_accs = [results['baseline']['val'], results['spatial']['val']]
    test_accs = [results['baseline']['test'], results['spatial']['test']]
    val_maes = [results['baseline']['val_mae'], results['spatial']['val_mae']]
    test_maes = [results['baseline']['test_mae'], results['spatial']['test_mae']]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, val_accs, width, label='Val Acc', color='#5B9BD5')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Acc', color='#ED7D31')

    ax.set_ylabel('Sum Accuracy (%)')
    ax.set_title('Spatial Attention Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(86, 96)

    # Add MAE on secondary axis (distinct colors for visibility)
    ax2 = ax.twinx()
    ax2.plot(x, val_maes, 'o--', color='#9467BD', label='Val MAE', markersize=8, linewidth=2)
    ax2.plot(x, test_maes, 's--', color='#2CA02C', label='Test MAE', markersize=8, linewidth=2)
    ax2.set_ylabel('MAE')
    ax2.set_ylim(0.25, 0.50)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=8)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/spatial_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUTPUT_DIR}/spatial_comparison.png')

if __name__ == '__main__':
    plot_width_comparison()
    plot_kernel_comparison()
    plot_sumloss_comparison()
    plot_augmentation_comparison()
    plot_spatial_comparison()
    print('Done!')
