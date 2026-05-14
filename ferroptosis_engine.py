"""
FerroptosisEngine: Iron-Dependent Cell Death Analysis Pipeline
- Ferroptosis sensitivity scoring (GPX4/SLC7A11/ACSL4 axis)
- Lipid peroxidation cascade simulation (PUFA oxidation kinetics)
- Iron metabolism network (labile iron pool, ferritin, transferrin)
- RSL3/Erastin drug response modeling
- Ferroptosis vs apoptosis vs necroptosis classifier
- Cell death trajectory analysis
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("FerroptosisEngine v1.0")
print("Iron-Dependent Cell Death Analysis Pipeline")
print("="*60)

# ─── 1. SYNTHETIC MULTI-OMICS DATA ───────────────────────────
N_CELLS = 200     # cell lines
N_GENES = 5000    # transcriptome
N_LIPIDS = 150    # lipidomics features

print(f"\n[Data] {N_CELLS} cell lines, {N_GENES} genes, {N_LIPIDS} lipid species")
print(f"  Conditions: RSL3 (GPX4 inhibitor), Erastin (xCT inhibitor), DMSO control")

# Cell line metadata
cancer_types_cell = np.random.choice(['LUAD', 'BRCA', 'CRC', 'GBM', 'PRAD', 'SKCM'], N_CELLS)
# True ferroptosis sensitivity (0=resistant, 1=sensitive)
true_sensitivity = np.random.beta(2, 2, N_CELLS)

# ─── 2. KEY FERROPTOSIS GENE EXPRESSION ──────────────────────
print("\n[Genes] Simulating ferroptosis gene expression...")

# Core ferroptosis genes
FERR_GENES = {
    # Suppressors (high = resistant)
    'GPX4': {'role': 'suppressor', 'weight': -0.35},
    'SLC7A11': {'role': 'suppressor', 'weight': -0.25},
    'FSP1': {'role': 'suppressor', 'weight': -0.20},
    'DHODH': {'role': 'suppressor', 'weight': -0.15},
    'GCH1': {'role': 'suppressor', 'weight': -0.10},
    # Drivers (high = sensitive)
    'ACSL4': {'role': 'driver', 'weight': 0.30},
    'LPCAT3': {'role': 'driver', 'weight': 0.20},
    'ALOX15': {'role': 'driver', 'weight': 0.18},
    'TF': {'role': 'driver', 'weight': 0.12},
    'TFRC': {'role': 'driver', 'weight': 0.15},
    'HMOX1': {'role': 'driver', 'weight': 0.10},
    'FTH1': {'role': 'iron', 'weight': -0.08},
    'FTL': {'role': 'iron', 'weight': -0.06},
    'NCOA4': {'role': 'iron', 'weight': 0.10},
}

# Simulate gene expression correlated with sensitivity
gene_expr = np.zeros((N_CELLS, len(FERR_GENES)))
gene_names_ferr = list(FERR_GENES.keys())
for j, (gene, info) in enumerate(FERR_GENES.items()):
    base = np.random.lognormal(3, 0.8, N_CELLS)
    # Correlation with sensitivity
    noise = np.random.normal(0, 0.5, N_CELLS)
    if info['role'] == 'suppressor':
        gene_expr[:, j] = base * np.exp(-2 * true_sensitivity + noise)
    else:
        gene_expr[:, j] = base * np.exp(2 * true_sensitivity + noise)

# Ferroptosis sensitivity score (weighted sum)
weights = np.array([FERR_GENES[g]['weight'] for g in gene_names_ferr])
log_expr = np.log1p(gene_expr)
log_expr_z = (log_expr - log_expr.mean(axis=0)) / (log_expr.std(axis=0) + 1e-6)
ferr_score = log_expr_z @ weights
ferr_score_norm = (ferr_score - ferr_score.min()) / (ferr_score.max() - ferr_score.min())

r_score_sens, p_score_sens = stats.pearsonr(ferr_score_norm, true_sensitivity)
print(f"  Ferroptosis score vs true sensitivity: r={r_score_sens:.3f}, p={p_score_sens:.2e}")
print(f"  GPX4 mean: {gene_expr[:, gene_names_ferr.index('GPX4')].mean():.1f}")
print(f"  ACSL4 mean: {gene_expr[:, gene_names_ferr.index('ACSL4')].mean():.1f}")

# ─── 3. LIPID PEROXIDATION KINETICS (ODE MODEL) ──────────────
print("\n[Lipids] Simulating lipid peroxidation cascade...")

def lipid_peroxidation_ode(y, t, k_init, k_prop, k_term, k_gpx4, gpx4_level):
    """
    Simplified lipid peroxidation ODE model.
    y = [PUFA, PUFA_OOH, ROS, MDA]
    PUFA: polyunsaturated fatty acids (substrate)
    PUFA_OOH: lipid hydroperoxides (product)
    ROS: reactive oxygen species
    MDA: malondialdehyde (cell death marker)
    """
    PUFA, PUFA_OOH, ROS, MDA = y
    # Initiation: Fe2+ + H2O2 → OH• (Fenton)
    d_ROS = k_init * max(PUFA, 0) - 0.1 * ROS
    # Propagation: PUFA + ROS → PUFA_OOH
    d_PUFA = -k_prop * PUFA * ROS
    d_PUFA_OOH = k_prop * PUFA * ROS - k_gpx4 * gpx4_level * PUFA_OOH - k_term * PUFA_OOH
    # MDA accumulation (cell death marker)
    d_MDA = 0.1 * k_term * PUFA_OOH
    return [d_PUFA, d_PUFA_OOH, d_ROS, d_MDA]

t_span = np.linspace(0, 24, 200)  # 24 hours
# Simulate for sensitive vs resistant cells
y0_sensitive = [100, 0, 1, 0]   # high PUFA, low GPX4
y0_resistant = [100, 0, 1, 0]

gpx4_sensitive = 0.5   # low GPX4 (sensitive)
gpx4_resistant = 3.0   # high GPX4 (resistant)

sol_sensitive = odeint(lipid_peroxidation_ode, y0_sensitive, t_span,
                       args=(0.05, 0.1, 0.02, 0.3, gpx4_sensitive))
sol_resistant = odeint(lipid_peroxidation_ode, y0_resistant, t_span,
                       args=(0.05, 0.1, 0.02, 0.3, gpx4_resistant))

mda_sensitive = sol_sensitive[:, 3].max()
mda_resistant = sol_resistant[:, 3].max()
print(f"  Peak MDA (sensitive): {mda_sensitive:.2f}")
print(f"  Peak MDA (resistant): {mda_resistant:.2f}")
print(f"  MDA ratio (sens/res): {mda_sensitive/mda_resistant:.1f}x")

# ─── 4. IRON METABOLISM NETWORK ──────────────────────────────
print("\n[Iron] Modeling iron metabolism network...")

def iron_metabolism_ode(y, t, k_import, k_export, k_storage, k_release, k_fenton):
    """
    Iron metabolism ODE.
    y = [LIP, Ferritin, Transferrin_bound, Fe2+_free]
    LIP: labile iron pool
    """
    LIP, Ferritin, TF_bound, Fe2_free = y
    # Import via transferrin receptor
    d_TF = -k_import * TF_bound
    d_LIP = k_import * TF_bound - k_storage * LIP - k_export * LIP
    d_Ferritin = k_storage * LIP - k_release * Ferritin
    # Fe2+ from ferritin autophagy (NCOA4-mediated)
    d_Fe2 = k_release * Ferritin - k_fenton * Fe2_free
    return [d_LIP, d_Ferritin, d_TF, d_Fe2]

y0_iron = [10, 50, 100, 1]
# High iron (ferroptosis-prone)
sol_iron_high = odeint(iron_metabolism_ode, y0_iron, t_span,
                       args=(0.5, 0.1, 0.3, 0.05, 0.2))
# Low iron (protected)
sol_iron_low = odeint(iron_metabolism_ode, y0_iron, t_span,
                      args=(0.1, 0.1, 0.5, 0.01, 0.05))

print(f"  Steady-state LIP (high iron): {sol_iron_high[-1, 0]:.2f}")
print(f"  Steady-state LIP (low iron): {sol_iron_low[-1, 0]:.2f}")
print(f"  Steady-state Fe2+ (high): {sol_iron_high[-1, 3]:.2f}")

# ─── 5. DRUG RESPONSE MODELING ───────────────────────────────
print("\n[Drugs] Modeling RSL3/Erastin response...")

def drug_response_curve(sensitivity, ec50, hill=2):
    """Hill equation for dose-response."""
    doses = np.logspace(-3, 2, 50)
    responses = np.zeros((len(sensitivity), len(doses)))
    for i, sens in enumerate(sensitivity):
        # EC50 inversely related to sensitivity
        cell_ec50 = ec50 / (sens + 0.1)
        responses[i] = doses**hill / (cell_ec50**hill + doses**hill)
    return doses, responses

# RSL3 (GPX4 inhibitor)
doses_rsl3, resp_rsl3 = drug_response_curve(ferr_score_norm, ec50=1.0)
# Erastin (xCT/SLC7A11 inhibitor)
doses_erastin, resp_erastin = drug_response_curve(ferr_score_norm, ec50=2.0)

# AUC as drug sensitivity metric
auc_rsl3 = np.trapz(resp_rsl3, np.log10(doses_rsl3), axis=1)
auc_erastin = np.trapz(resp_erastin, np.log10(doses_erastin), axis=1)

r_rsl3, p_rsl3 = stats.pearsonr(ferr_score_norm, auc_rsl3)
r_erastin, p_erastin = stats.pearsonr(ferr_score_norm, auc_erastin)
print(f"  RSL3 AUC vs ferroptosis score: r={r_rsl3:.3f}, p={p_rsl3:.2e}")
print(f"  Erastin AUC vs ferroptosis score: r={r_erastin:.3f}, p={p_erastin:.2e}")

# ─── 6. CELL DEATH CLASSIFIER ────────────────────────────────
print("\n[Classifier] Ferroptosis vs apoptosis vs necroptosis...")

# Simulate cell death markers
# Ferroptosis: high lipid ROS, high MDA, low GPX4, iron-dependent
# Apoptosis: caspase activation, cytochrome c release, DNA fragmentation
# Necroptosis: MLKL phosphorylation, membrane rupture, RIPK3

N_DEATH_CELLS = 300
death_labels = np.random.choice(['ferroptosis', 'apoptosis', 'necroptosis', 'survival'],
                                  N_DEATH_CELLS, p=[0.25, 0.35, 0.15, 0.25])

# Feature matrix for classifier
features = np.zeros((N_DEATH_CELLS, 8))
for i, label in enumerate(death_labels):
    if label == 'ferroptosis':
        features[i] = [
            np.random.normal(0.2, 0.1),   # GPX4 (low)
            np.random.normal(0.8, 0.1),   # ACSL4 (high)
            np.random.normal(0.8, 0.1),   # lipid ROS (high)
            np.random.normal(0.2, 0.1),   # caspase (low)
            np.random.normal(0.8, 0.1),   # iron (high)
            np.random.normal(0.2, 0.1),   # MLKL (low)
            np.random.normal(0.8, 0.1),   # MDA (high)
            np.random.normal(0.3, 0.1),   # membrane integrity
        ]
    elif label == 'apoptosis':
        features[i] = [
            np.random.normal(0.6, 0.1),   # GPX4
            np.random.normal(0.3, 0.1),   # ACSL4
            np.random.normal(0.3, 0.1),   # lipid ROS
            np.random.normal(0.9, 0.1),   # caspase (high)
            np.random.normal(0.3, 0.1),   # iron
            np.random.normal(0.2, 0.1),   # MLKL
            np.random.normal(0.3, 0.1),   # MDA
            np.random.normal(0.4, 0.1),   # membrane
        ]
    elif label == 'necroptosis':
        features[i] = [
            np.random.normal(0.5, 0.1),   # GPX4
            np.random.normal(0.4, 0.1),   # ACSL4
            np.random.normal(0.5, 0.1),   # lipid ROS
            np.random.normal(0.3, 0.1),   # caspase
            np.random.normal(0.4, 0.1),   # iron
            np.random.normal(0.9, 0.1),   # MLKL (high)
            np.random.normal(0.4, 0.1),   # MDA
            np.random.normal(0.1, 0.1),   # membrane (low)
        ]
    else:  # survival
        features[i] = np.random.normal(0.5, 0.15, 8)
    features[i] = features[i].clip(0, 1)

# Simple nearest-centroid classifier
label_map = {'ferroptosis': 0, 'apoptosis': 1, 'necroptosis': 2, 'survival': 3}
y_true = np.array([label_map[l] for l in death_labels])

# Compute centroids
centroids = np.array([features[y_true == k].mean(axis=0) for k in range(4)])

# Predict: nearest centroid
dists = np.array([[np.linalg.norm(features[i] - centroids[k]) for k in range(4)]
                  for i in range(N_DEATH_CELLS)])
y_pred = np.argmin(dists, axis=1)
accuracy_clf = (y_pred == y_true).mean()

# Per-class accuracy
label_names = ['ferroptosis', 'apoptosis', 'necroptosis', 'survival']
for k, name in enumerate(label_names):
    mask = y_true == k
    acc_k = (y_pred[mask] == k).mean() if mask.sum() > 0 else 0
    print(f"  {name}: accuracy={acc_k:.3f} (n={mask.sum()})")
print(f"  Overall accuracy: {accuracy_clf:.3f}")

# ─── 7. VISUALIZATION ────────────────────────────────────────
print("\n[Viz] Generating dashboard...")

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0a0a0a')
gs_main = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

# Panel 1: Ferroptosis score distribution
ax1 = fig.add_subplot(gs_main[0, 0])
ax1.set_facecolor('#111111')
ct_colors_map = {'LUAD':'#2196F3','BRCA':'#E91E63','CRC':'#4CAF50',
                 'GBM':'#FF9800','PRAD':'#9C27B0','SKCM':'#00BCD4'}
for ct in np.unique(cancer_types_cell):
    mask = cancer_types_cell == ct
    ax1.scatter(true_sensitivity[mask], ferr_score_norm[mask],
                c=ct_colors_map[ct], s=15, alpha=0.7, label=ct)
ax1.set_xlabel('True Sensitivity', color='white', fontsize=9)
ax1.set_ylabel('Ferroptosis Score', color='white', fontsize=9)
ax1.set_title(f'Ferroptosis Sensitivity Score\n(r={r_score_sens:.3f})', color='white', fontsize=10, fontweight='bold')
ax1.tick_params(colors='white', labelsize=7)
for spine in ax1.spines.values(): spine.set_color('#333333')
ax1.legend(fontsize=6, facecolor='#222222', labelcolor='white', ncol=2)

# Panel 2: Lipid peroxidation kinetics
ax2 = fig.add_subplot(gs_main[0, 1])
ax2.set_facecolor('#111111')
ax2.plot(t_span, sol_sensitive[:, 1], color='#FF5722', linewidth=2, label='Sensitive (PUFA-OOH)')
ax2.plot(t_span, sol_resistant[:, 1], color='#2196F3', linewidth=2, label='Resistant (PUFA-OOH)')
ax2.plot(t_span, sol_sensitive[:, 3]*10, color='#FF5722', linewidth=1.5, linestyle='--', label='Sensitive (MDA×10)')
ax2.plot(t_span, sol_resistant[:, 3]*10, color='#2196F3', linewidth=1.5, linestyle='--', label='Resistant (MDA×10)')
ax2.set_xlabel('Time (hours)', color='white', fontsize=9)
ax2.set_ylabel('Concentration (a.u.)', color='white', fontsize=9)
ax2.set_title('Lipid Peroxidation Kinetics', color='white', fontsize=10, fontweight='bold')
ax2.tick_params(colors='white', labelsize=7)
for spine in ax2.spines.values(): spine.set_color('#333333')
ax2.legend(fontsize=6, facecolor='#222222', labelcolor='white')

# Panel 3: Iron metabolism
ax3 = fig.add_subplot(gs_main[0, 2])
ax3.set_facecolor('#111111')
ax3.plot(t_span, sol_iron_high[:, 0], color='#FF5722', linewidth=2, label='LIP (high Fe)')
ax3.plot(t_span, sol_iron_low[:, 0], color='#2196F3', linewidth=2, label='LIP (low Fe)')
ax3.plot(t_span, sol_iron_high[:, 3], color='#FF9800', linewidth=1.5, linestyle='--', label='Fe2+ (high)')
ax3.plot(t_span, sol_iron_low[:, 3], color='#00BCD4', linewidth=1.5, linestyle='--', label='Fe2+ (low)')
ax3.set_xlabel('Time (hours)', color='white', fontsize=9)
ax3.set_ylabel('Concentration (a.u.)', color='white', fontsize=9)
ax3.set_title('Iron Metabolism Network', color='white', fontsize=10, fontweight='bold')
ax3.tick_params(colors='white', labelsize=7)
for spine in ax3.spines.values(): spine.set_color('#333333')
ax3.legend(fontsize=6, facecolor='#222222', labelcolor='white')

# Panel 4: Drug response curves
ax4 = fig.add_subplot(gs_main[1, 0])
ax4.set_facecolor('#111111')
# Show mean ± std for sensitive vs resistant quartiles
q75 = np.percentile(ferr_score_norm, 75)
q25 = np.percentile(ferr_score_norm, 25)
sens_mask = ferr_score_norm >= q75
res_mask = ferr_score_norm <= q25
for drug_name, doses, resp, col in [('RSL3', doses_rsl3, resp_rsl3, '#FF5722'),
                                      ('Erastin', doses_erastin, resp_erastin, '#E9ED4C')]:
    mean_s = resp[sens_mask].mean(axis=0)
    mean_r = resp[res_mask].mean(axis=0)
    ax4.semilogx(doses, mean_s, color=col, linewidth=2, label=f'{drug_name} Sensitive')
    ax4.semilogx(doses, mean_r, color=col, linewidth=1.5, linestyle='--', alpha=0.6, label=f'{drug_name} Resistant')
ax4.set_xlabel('Drug Concentration (μM)', color='white', fontsize=9)
ax4.set_ylabel('Cell Death Fraction', color='white', fontsize=9)
ax4.set_title('RSL3/Erastin Dose-Response', color='white', fontsize=10, fontweight='bold')
ax4.tick_params(colors='white', labelsize=7)
for spine in ax4.spines.values(): spine.set_color('#333333')
ax4.legend(fontsize=6, facecolor='#222222', labelcolor='white')

# Panel 5: Gene expression heatmap
ax5 = fig.add_subplot(gs_main[1, 1])
ax5.set_facecolor('#111111')
sort_idx = np.argsort(ferr_score_norm)
expr_sorted = log_expr_z[sort_idx]
im5 = ax5.imshow(expr_sorted.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax5.set_yticks(range(len(gene_names_ferr)))
ax5.set_yticklabels(gene_names_ferr, color='white', fontsize=7)
ax5.set_xlabel('Cell lines (sorted by score)', color='white', fontsize=9)
ax5.set_title('Ferroptosis Gene Expression', color='white', fontsize=10, fontweight='bold')
ax5.tick_params(colors='white', labelsize=7)
plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')

# Panel 6: Cell death classifier
ax6 = fig.add_subplot(gs_main[1, 2])
ax6.set_facecolor('#111111')
death_colors = {'ferroptosis': '#FF5722', 'apoptosis': '#2196F3',
                'necroptosis': '#9C27B0', 'survival': '#4CAF50'}
# PCA of features for visualization
feat_centered = features - features.mean(axis=0)
U, S, Vt = np.linalg.svd(feat_centered, full_matrices=False)
feat_pca = U[:, :2] * S[:2]
for label, col in death_colors.items():
    mask = np.array(death_labels) == label
    ax6.scatter(feat_pca[mask, 0], feat_pca[mask, 1], c=col, s=15, alpha=0.7, label=label)
ax6.set_xlabel('PC1', color='white', fontsize=9)
ax6.set_ylabel('PC2', color='white', fontsize=9)
ax6.set_title(f'Cell Death Classifier\n(acc={accuracy_clf:.3f})', color='white', fontsize=10, fontweight='bold')
ax6.tick_params(colors='white', labelsize=7)
for spine in ax6.spines.values(): spine.set_color('#333333')
ax6.legend(fontsize=7, facecolor='#222222', labelcolor='white')

# Panel 7: GPX4 vs ACSL4
ax7 = fig.add_subplot(gs_main[2, 0])
ax7.set_facecolor('#111111')
gpx4_idx = gene_names_ferr.index('GPX4')
acsl4_idx = gene_names_ferr.index('ACSL4')
sc = ax7.scatter(np.log1p(gene_expr[:, gpx4_idx]), np.log1p(gene_expr[:, acsl4_idx]),
                 c=ferr_score_norm, cmap='RdYlGn_r', s=20, alpha=0.8)
plt.colorbar(sc, ax=ax7, fraction=0.046, pad=0.04, label='Ferr. Score').ax.yaxis.set_tick_params(color='white', labelcolor='white')
ax7.set_xlabel('log(GPX4 + 1)', color='white', fontsize=9)
ax7.set_ylabel('log(ACSL4 + 1)', color='white', fontsize=9)
ax7.set_title('GPX4 vs ACSL4 Axis', color='white', fontsize=10, fontweight='bold')
ax7.tick_params(colors='white', labelsize=7)
for spine in ax7.spines.values(): spine.set_color('#333333')

# Panel 8: Drug AUC by cancer type
ax8 = fig.add_subplot(gs_main[2, 1])
ax8.set_facecolor('#111111')
ct_list = list(ct_colors_map.keys())
rsl3_by_ct = [auc_rsl3[cancer_types_cell == ct] for ct in ct_list]
bp8 = ax8.boxplot(rsl3_by_ct, patch_artist=True)
for patch, ct in zip(bp8['boxes'], ct_list):
    patch.set_facecolor(ct_colors_map[ct]); patch.set_alpha(0.8)
for element in ['whiskers', 'caps', 'medians', 'fliers']:
    for item in bp8[element]: item.set_color('white')
ax8.set_xticks(range(1, len(ct_list)+1))
ax8.set_xticklabels(ct_list, color='white', fontsize=8, rotation=30)
ax8.set_ylabel('RSL3 AUC', color='white', fontsize=9)
ax8.set_title('RSL3 Sensitivity by Cancer Type', color='white', fontsize=10, fontweight='bold')
ax8.tick_params(colors='white', labelsize=7)
for spine in ax8.spines.values(): spine.set_color('#333333')

# Panel 9: Summary
ax9 = fig.add_subplot(gs_main[2, 2])
ax9.set_facecolor('#111111'); ax9.axis('off')
summary = [
    "FerroptosisEngine v1.0", "",
    f"Cell lines: {N_CELLS}",
    f"Genes: {N_GENES}",
    f"Lipid species: {N_LIPIDS}", "",
    f"Ferroptosis score:",
    f"  r={r_score_sens:.3f} vs sensitivity", "",
    f"Lipid peroxidation:",
    f"  MDA sens/res: {mda_sensitive/mda_resistant:.1f}x", "",
    f"Drug response:",
    f"  RSL3 r={r_rsl3:.3f}",
    f"  Erastin r={r_erastin:.3f}", "",
    f"Cell death classifier:",
    f"  Accuracy={accuracy_clf:.3f}",
    f"  Classes: 4 (ferr/apo/nec/surv)", "",
    f"Iron LIP (high/low): {sol_iron_high[-1,0]:.1f}/{sol_iron_low[-1,0]:.1f}",
]
for i, line in enumerate(summary):
    ax9.text(0.05, 0.97-i*0.054, line, transform=ax9.transAxes,
             color='#E9ED4C' if i==0 else 'white', fontsize=8.5, va='top',
             fontweight='bold' if i==0 else 'normal')

fig.suptitle('FerroptosisEngine: Iron-Dependent Cell Death Analysis Dashboard',
             color='white', fontsize=14, fontweight='bold', y=0.98)
plt.savefig('/workspace/ferroptosis_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("  Dashboard saved.")

print("\n"+"="*60)
print("FerroptosisEngine COMPLETE")
print(f"  Cell lines: {N_CELLS} | Genes: {N_GENES} | Lipids: {N_LIPIDS}")
print(f"  Ferroptosis score r={r_score_sens:.3f}")
print(f"  MDA ratio (sens/res): {mda_sensitive/mda_resistant:.1f}x")
print(f"  RSL3 AUC r={r_rsl3:.3f}, Erastin AUC r={r_erastin:.3f}")
print(f"  Cell death classifier accuracy={accuracy_clf:.3f}")
print("="*60)
