# 📈 All Model Convergences for a Specific Slide
# 10 Models Convergence in Subplots (5 columns, 2 rows for 10 models)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Choose which slide to analyze (0-indexed, so slide 1 = index 0)
slide_to_analyze = 0  # Change this to analyze different slides (0-9)

# Define visual styling - Fixed RGB colors for each optimizer
marker_styles = ['o', 's', '^']  # GP Minimize, Forest Minimize, Dummy Minimize
optimizer_colors = ['red', 'blue', 'green']  # Fixed colors for GP-EI, Forest, Dummy

# Define all model results in order
model_results = [
    (res_all_gp_ei_ET, res_all_forest_ET, res_all_dummy_ET, 'ET'),
    (res_all_gp_ei_RP, res_all_forest_RP, res_all_dummy_RP, 'RP'),
    (res_all_gp_ei_RotF, res_all_forest_RotF, res_all_dummy_RotF, 'RotF'),
    (res_all_gp_ei_RRotF, res_all_forest_RRotF, res_all_dummy_RRotF, 'RRotF'),
    (res_all_gp_ei_AdaB, res_all_forest_AdaB, res_all_dummy_AdaB, 'AdaB'),
    (res_all_gp_ei_GBM, res_all_forest_GBM, res_all_dummy_GBM, 'GBM'),
    (res_all_gp_ei_LB, res_all_forest_LB, res_all_dummy_LB, 'LB'),
    (res_all_gp_ei_XGB, res_all_forest_XGB, res_all_dummy_XGB, 'XGB'),
    (res_all_gp_ei_CAT, res_all_forest_CAT, res_all_dummy_CAT, 'CatB'),
    (res_all_gp_ei_LGB, res_all_forest_LGB, res_all_dummy_LGB, 'LGB')
]

n_models = len(model_results)  # 10 models
n_cols = 5  # 5 columns
n_rows = 2  # 2 rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 12))

for model_idx, (gp_results, forest_results, dummy_results, model_name) in enumerate(model_results):
    row = model_idx // n_cols
    col = model_idx % n_cols
    ax = axes[row, col]
    
    # Extract convergence data for the specific slide
    gp_vals = np.minimum.accumulate(gp_results[slide_to_analyze].func_vals)
    forest_vals = np.minimum.accumulate(forest_results[slide_to_analyze].func_vals)
    dummy_vals = np.minimum.accumulate(dummy_results[slide_to_analyze].func_vals)
    
    # Plot three optimizers with fixed RGB colors (no legend in individual subplots)
    ax.plot(gp_vals, color=optimizer_colors[0], marker=marker_styles[0], 
            linewidth=4, markersize=5, markevery=max(1, len(gp_vals)//8), alpha=0.8)
    ax.plot(forest_vals, color=optimizer_colors[1], marker=marker_styles[1], 
            linewidth=2, markersize=5, markevery=max(1, len(forest_vals)//8), alpha=0.8)
    ax.plot(dummy_vals, color=optimizer_colors[2], marker=marker_styles[2], 
            linewidth=2, markersize=5, markevery=max(1, len(dummy_vals)//8), alpha=0.8)
    
    ax.set_xlabel("Number of Calls", fontsize=30)
    ax.set_ylabel("Min Objective Value", fontsize=30)
    ax.set_title(f"{model_name}", fontsize=30, fontweight='bold')
    ax.tick_params(axis='both', labelsize=22)
    ax.grid(True, alpha=0.3)

# Create global legend at the bottom of the figure
legend_elements = [
    plt.Line2D([0], [0], color=optimizer_colors[0], marker=marker_styles[0], 
               linewidth=4, markersize=8, label='GP', alpha=0.8),
    plt.Line2D([0], [0], color=optimizer_colors[1], marker=marker_styles[1], 
               linewidth=2, markersize=8, label='Forest', alpha=0.8),
    plt.Line2D([0], [0], color=optimizer_colors[2], marker=marker_styles[2], 
               linewidth=2, markersize=8, label='Dummy', alpha=0.8)
]

# Add the legend below all subplots
fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
          fontsize=35, bbox_to_anchor=(0.5, -0.09), frameon=True, 
          fancybox=True, shadow=True)

# Add a main title for the entire figure
#fig.suptitle(f'Bayesian Optimization Convergence Comparison - Slide {slide_to_analyze + 1}', 
             #fontsize=40, fontweight='bold', y=1.02)

plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.1)  # Make room for title and legend
plt.savefig(f"/kaggle/working/All_Models_Convergence_Slide_{slide_to_analyze + 1}.tiff", 
            bbox_inches='tight', dpi=300)
plt.show()

# Optional: Create a summary comparison
print(f"\n📊 Summary for Slide {slide_to_analyze + 1}:")
print("="*60)
for model_idx, (gp_results, forest_results, dummy_results, model_name) in enumerate(model_results):
    gp_best = np.min(gp_results[slide_to_analyze].func_vals)
    forest_best = np.min(forest_results[slide_to_analyze].func_vals)
    dummy_best = np.min(dummy_results[slide_to_analyze].func_vals)
    
    print(f"{model_name:18s} - GP: {gp_best:.4f}, Forest: {forest_best:.4f}, Dummy: {dummy_best:.4f}")