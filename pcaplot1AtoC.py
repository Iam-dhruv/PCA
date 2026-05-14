import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


labels = pd.read_csv('data/class.tsv', header=None, names=['ER_Status'])
cols_df = pd.read_csv('data/columns.tsv', sep='\t', comment='#')
cols_df['GeneSymbol'] = cols_df['GeneSymbol'].astype(str).str.strip()

expr_df = pd.read_csv('data/filtered.tsv', sep='\t')
expr_df.columns = expr_df.columns.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

id_xbp1 = cols_df.loc[cols_df['GeneSymbol'] == 'XBP1', 'ID'].values[0]
id_gata3 = cols_df.loc[cols_df['GeneSymbol'] == 'GATA3', 'ID'].values[0]

col_xbp1 = str(int(id_xbp1)).strip()
col_gata3 = str(int(id_gata3)).strip()

X = expr_df[[col_gata3, col_xbp1]]

mask_er_neg = (labels['ER_Status'] == 0)
mask_er_pos = (labels['ER_Status'] == 1)

plt.figure(figsize=(6, 5))
plt.scatter(X[mask_er_neg][col_gata3], X[mask_er_neg][col_xbp1], c='black', label='ER- (0)', s=15)
plt.scatter(X[mask_er_pos][col_gata3], X[mask_er_pos][col_xbp1], c='red', label='ER+ (1)', s=15)
plt.xlabel('GATA3')
plt.ylabel('XBP1')
plt.title('Figure 1a: 2-Gene Scatter Plot')

fig_1a_path = os.path.join(os.getcwd(), 'figure_1a.png')
plt.savefig(fig_1a_path, dpi=300, bbox_inches='tight')
print(f"Saved Figure 1a to: {fig_1a_path}")
plt.close()


pca_2d = PCA(n_components=2)
pca_2d.fit(X)

plt.figure(figsize=(6, 5))
plt.scatter(X[mask_er_neg][col_gata3], X[mask_er_neg][col_xbp1], c='black', s=15)
plt.scatter(X[mask_er_pos][col_gata3], X[mask_er_pos][col_xbp1], c='red', s=15)

mean_gata3, mean_xbp1 = X.mean()
for length, vector in zip(pca_2d.explained_variance_, pca_2d.components_):
    v = vector * 2 * np.sqrt(length) 
    plt.annotate('', xy=(mean_gata3 + v[0], mean_xbp1 + v[1]), xytext=(mean_gata3, mean_xbp1),
                 arrowprops=dict(arrowstyle="->", linewidth=1.5))

plt.xlabel('GATA3')
plt.ylabel('XBP1')
plt.title('Figure 1b: Principal Component Axes')

fig_1b_path = os.path.join(os.getcwd(), 'figure_1b.png')
plt.savefig(fig_1b_path, dpi=300, bbox_inches='tight')
print(f"Saved Figure 1b to: {fig_1b_path}")
plt.close()


X_pc1 = pca_2d.transform(X)[:, 0]

plt.figure(figsize=(6, 3))
plt.scatter(X_pc1[mask_er_neg], np.zeros(sum(mask_er_neg)), c='black', s=15)
plt.scatter(X_pc1[mask_er_pos], np.ones(sum(mask_er_pos)), c='red', s=15)
plt.xlabel('Projection onto PC1')
plt.yticks([0, 1], ['ER-', 'ER+'])
plt.ylim(-0.5, 1.5)
plt.title('Figure 1c: 1D Projection onto PC1')

fig_1c_path = os.path.join(os.getcwd(), 'figure_1c.png')
plt.savefig(fig_1c_path, dpi=300, bbox_inches='tight')
print(f"Saved Figure 1c to: {fig_1c_path}")
plt.close()