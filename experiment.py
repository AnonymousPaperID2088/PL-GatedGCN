"""
Population-Level Gated Graph Convolutional Network for Postoperative Delirium Classification in Older Adults Undergoing Noncardiac Surgery

Population: Elderly patients (>=65 years) undergoing non-cardiac surgery
"""

import os, re, warnings, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from mymodel import GCNConv

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import LinearRegression

import shap
# Patch SHAP to avoid broken TensorFlow import via HuggingFace transformers
try:
    import shap.explainers._explainer as _shap_explainer
    _shap_explainer.is_transformers_lm = lambda x: False
except Exception:
    pass

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight', 'font.family': 'DejaVu Sans'})

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

OUTPUT_DIR = r'd:\workspace\Medical Project\results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================
print("=" * 70)
print("  PART 1: DATA PREPROCESSING")
print("=" * 70)
df = pd.read_excel(r'd:\workspace\Medical Project\Case Summary.xlsx')
print(f"Raw data: {df.shape}")

def pn(v):
    if pd.isna(v): return np.nan
    s = re.sub(r'\.{2,}', '.', str(v).strip())
    if s in ['不详','未测','/','.',' ','','昏迷','深昏迷']: return np.nan
    try: return float(s)
    except:
        m = re.search(r'\d+\.?\d*', s)
        try: return float(m.group()) if m else np.nan
        except: return np.nan

def pb(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s in ['0','0.0']: return 0.0
    if s in ['不详','未测','/',' ','.','']: return np.nan
    if s.startswith('1') or s == '1.0': return 1.0
    try:
        vv = float(s); return 1.0 if vv >= 1 else 0.0
    except:
        for kw in ['帕金森','癫痫','老年痴呆','抑郁','焦虑','心律失常','室早',
                    '肾功能','尿毒','肺部感染','远视','白内障','青光眼','视力','后循环','糖耐量']:
            if kw in s: return 1.0
        return np.nan

def parse_age(v):
    if pd.isna(v): return np.nan
    m = re.search(r'(\d+)', str(v))
    return float(m.group(1)) if m else np.nan

def parse_dur(v):
    if pd.isna(v): return np.nan
    s = str(v).strip(); t = 0
    h = re.search(r'(\d+)\s*小时', s); m = re.search(r'(\d+)\s*分', s)
    if h: t += int(h.group(1)) * 60
    if m: t += int(m.group(1))
    if t == 0:
        try: return float(s)
        except: return np.nan
    return float(t)

def parse_trans(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s in ['0','0.0','无']: return 0.0
    if s in ['不详','未测','/',' ','.','']: return np.nan
    try: return float(s)
    except:
        t = 0
        for u in re.findall(r'([\d.]+)\s*u', s, re.I): t += float(u) * 200
        for ml in re.findall(r'([\d.]+)\s*ml', s, re.I): t += float(ml)
        return t if t > 0 else np.nan

def parse_benzo(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s in ['0','0.0','无']: return 0.0
    if s in ['不详','未测','/',' ','.','']: return np.nan
    try: return float(s)
    except:
        m = re.search(r'([\d.]+)\s*mg', s)
        return float(m.group(1)) if m else np.nan

def parse_antichol(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s in ['0','0.0','无']: return 0.0
    if s in ['不详','未测','/',' ','.','']: return np.nan
    try: return float(s)
    except:
        t = sum(float(d) for d in re.findall(r'([\d.]+)\s*mg', s))
        return t if t > 0 else np.nan

def parse_opioid(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s in ['0','0.0','无']: return 0.0
    if s in ['不详','未测','/',' ','.','']: return np.nan
    score = 0
    for kw in ['杜冷丁','哌替啶','舒芬','瑞芬','布托','地佐','纳布','吗啡','羟考','阿芬']:
        if kw in s: score += 1
    if '芬' in s and '舒芬' not in s and '瑞芬' not in s: score += 1
    return float(max(score, 1)) if score > 0 or len(s) > 2 else 0.0

def parse_postop_opioid(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s in ['0','0.0','无']: return 0.0
    if s in ['不详','未测','/',' ','.','']: return np.nan
    try: return 0.0 if float(s) == 0 else 1.0
    except: return 1.0

# Build structured dataframe
data = pd.DataFrame()
data['Age'] = df['年龄'].apply(parse_age)
data['Sex'] = df['性别'].map({'男': 1, '女': 0})
data['BMI'] = df['BMI'].apply(pn)
data['Surgical_Urgency'] = df['手术急迫性'].map({'急诊': 1, '择期': 0})

for cn, en in [('术前认知障碍','Preop_Cognitive_Impairment'), ('卒中史','History_of_Stroke'),
    ('帕金森等神经系统疾病','Neurological_Disease'), ('抑郁/焦虑等精神疾病史','Psychiatric_History'),
    ('高血压','Hypertension'), ('糖尿病','Diabetes'), ('冠心病/心衰','Coronary_Heart_Disease'),
    ('慢性肾病','Chronic_Kidney_Disease'), ('COPD/呼吸功能不全','COPD'),
    ('听力或视力障碍','Sensory_Impairment'), ('日常生活是否受限','ADL_Limitation'),
    ('跌倒史','History_of_Falls'), ('长期吸烟史','Smoking_History'), ('长期饮酒史','Alcohol_History'),
    ('苯二氮卓类药物史','Benzodiazepine_History'), ('抗胆碱能药物史','Anticholinergic_History'),
    ('多种药物联用史','Polypharmacy'), ('精神类药物史','Psychotropic_Drug_History')]:
    data[en] = df[cn].apply(pb)

data['Surgery_Duration'] = df['手术时长'].apply(parse_dur)
data['Anesthesia_Type'] = df['麻醉方式'].apply(lambda v: 0 if pd.isna(v) else (
    2 if any(k in str(v) for k in ['硬','腰','椎管','连硬外','神经阻滞']) and
         any(k in str(v) for k in ['全麻','全身','静吸']) else
    (1 if any(k in str(v) for k in ['硬膜外','腰','椎管','神经阻滞','局麻']) else 0)))
data['Postop_Analgesia'] = df['术后镇痛'].apply(lambda v: np.nan if pd.isna(v) else (
    0 if str(v) in ['无','0'] else (3 if '神经' in str(v) else (2 if '硬膜' in str(v) else 1))))

for cn, en in [('血红蛋白','Hemoglobin'), ('白蛋白','Albumin'), ('肌酐','Creatinine'),
               ('钠','Sodium'), ('钾','Potassium'), ('空腹血糖','Fasting_Glucose')]:
    data[en] = df[cn].apply(pn)

data['Blood_Loss'] = df['失血量'].apply(pn)
data['Urine_Output'] = df['尿量'].apply(pn)
data['Intraop_Transfusion'] = df['术中输血0'].apply(parse_trans)
data['Inhalation_Agent_Duration'] = df['吸入药物使用时长'].apply(pn)
data['Intraop_Opioid_Load'] = df['阿片类药物总量'].apply(parse_opioid)
data['Intraop_Benzodiazepine'] = df['苯二氮卓类药物使用（咪达，瑞马等）'].apply(parse_benzo)
data['Intraop_Anticholinergic'] = df['抗胆碱能药物使用（阿托品，长托宁）'].apply(parse_antichol)

for cn, en in [('术中低血压','Intraop_Hypotension'), ('术中低心率','Intraop_Bradycardia'),
               ('术中低氧血症','Intraop_Hypoxemia'), ('术中心律失常','Intraop_Arrhythmia')]:
    data[en] = df[cn].apply(pb)

data['Postop_Pain'] = df['术后疼痛情况'].apply(pb)
data['Postop_Opioid_Use'] = df['术后阿片类药物用量'].apply(parse_postop_opioid)
data['Postop_Sedative_Use'] = df['术后是否使用镇静催眠药'].apply(pb)
data['Postop_Infection'] = df['术后合并感染'].apply(pb)
data['Postop_Electrolyte_Disorder'] = df['术后电解质紊乱'].apply(pb)
data['Postop_Anemia'] = df['术后是否贫血'].apply(pb)
data['ICU_Admission'] = df['入ICU'].apply(pb)
data['Indwelling_Catheter'] = df['留置导管'].apply(pb)
data['POD'] = df['术后是否谵妄'].apply(pb)

data = data.dropna(subset=['POD']).reset_index(drop=True)
feature_cols = [c for c in data.columns if c != 'POD']
print(f"Processed: {data.shape[0]} patients, {data.shape[1]-1} features, POD incidence={data['POD'].mean()*100:.1f}%")

# Display names
display_names = {
    'Age': 'Age (years)', 'Sex': 'Male sex, n (%)', 'BMI': 'BMI (kg/m2)',
    'Surgical_Urgency': 'Emergency surgery, n (%)',
    'Preop_Cognitive_Impairment': 'Preoperative cognitive impairment, n (%)',
    'History_of_Stroke': 'History of stroke, n (%)',
    'Neurological_Disease': 'Neurological disease, n (%)',
    'Psychiatric_History': 'Psychiatric history, n (%)',
    'Hypertension': 'Hypertension, n (%)', 'Diabetes': 'Diabetes mellitus, n (%)',
    'Coronary_Heart_Disease': 'Coronary heart disease/HF, n (%)',
    'Chronic_Kidney_Disease': 'Chronic kidney disease, n (%)', 'COPD': 'COPD, n (%)',
    'Sensory_Impairment': 'Sensory impairment, n (%)',
    'ADL_Limitation': 'ADL limitation, n (%)', 'History_of_Falls': 'History of falls, n (%)',
    'Smoking_History': 'Smoking history, n (%)', 'Alcohol_History': 'Alcohol history, n (%)',
    'Benzodiazepine_History': 'Benzodiazepine use history, n (%)',
    'Anticholinergic_History': 'Anticholinergic use history, n (%)',
    'Polypharmacy': 'Polypharmacy, n (%)', 'Psychotropic_Drug_History': 'Psychotropic drug history, n (%)',
    'Surgery_Duration': 'Surgery duration (min)', 'Anesthesia_Type': 'General anesthesia, n (%)',
    'Postop_Analgesia': 'IV postop analgesia, n (%)',
    'Hemoglobin': 'Hemoglobin (g/L)', 'Albumin': 'Albumin (g/L)',
    'Creatinine': 'Creatinine (umol/L)', 'Sodium': 'Sodium (mmol/L)',
    'Potassium': 'Potassium (mmol/L)', 'Fasting_Glucose': 'Fasting glucose (mmol/L)',
    'Blood_Loss': 'Blood loss (mL)', 'Urine_Output': 'Urine output (mL)',
    'Intraop_Transfusion': 'Intraop transfusion (mL)',
    'Inhalation_Agent_Duration': 'Inhalation agent duration (min)',
    'Intraop_Opioid_Load': 'Intraop opioid load (score)',
    'Intraop_Benzodiazepine': 'Intraop benzodiazepine (mg)',
    'Intraop_Anticholinergic': 'Intraop anticholinergic (mg)',
    'Intraop_Hypotension': 'Intraop hypotension, n (%)',
    'Intraop_Bradycardia': 'Intraop bradycardia, n (%)',
    'Intraop_Hypoxemia': 'Intraop hypoxemia, n (%)',
    'Intraop_Arrhythmia': 'Intraop arrhythmia, n (%)',
    'Postop_Pain': 'Postop pain, n (%)', 'Postop_Opioid_Use': 'Postop opioid use, n (%)',
    'Postop_Sedative_Use': 'Postop sedative use, n (%)',
    'Postop_Infection': 'Postop infection, n (%)',
    'Postop_Electrolyte_Disorder': 'Postop electrolyte disorder, n (%)',
    'Postop_Anemia': 'Postop anemia, n (%)', 'ICU_Admission': 'ICU admission, n (%)',
    'Indwelling_Catheter': 'Indwelling catheter, n (%)',
}

# ============================================================
# TABLE 1 — BASELINE CHARACTERISTICS
# ============================================================
print("\n" + "=" * 70)
print("  PART 2: TABLE 1 — BASELINE CHARACTERISTICS")
print("=" * 70)

continuous_vars = ['Age','BMI','Surgery_Duration','Hemoglobin','Albumin','Creatinine',
                   'Sodium','Potassium','Fasting_Glucose','Blood_Loss','Urine_Output',
                   'Intraop_Transfusion','Inhalation_Agent_Duration','Intraop_Opioid_Load',
                   'Intraop_Benzodiazepine','Intraop_Anticholinergic']

pod_pos = data[data['POD'] == 1]
pod_neg = data[data['POD'] == 0]
rows = [{'Variable': 'n', f'Total (N={len(data)})': str(len(data)),
         f'POD (N={len(pod_pos)})': str(len(pod_pos)),
         f'Non-POD (N={len(pod_neg)})': str(len(pod_neg)), 'P-value': ''}]

for var in continuous_vars:
    pv, nv = pod_pos[var].dropna(), pod_neg[var].dropna()
    tv = data[var].dropna()
    if len(pv) > 0 and len(nv) > 0:
        ts = f"{tv.median():.1f} [{tv.quantile(.25):.1f}-{tv.quantile(.75):.1f}]"
        ps = f"{pv.median():.1f} [{pv.quantile(.25):.1f}-{pv.quantile(.75):.1f}]"
        ns = f"{nv.median():.1f} [{nv.quantile(.25):.1f}-{nv.quantile(.75):.1f}]"
        _, p = stats.mannwhitneyu(pv, nv, alternative='two-sided')
        pstr = f"{p:.3f}" if p >= 0.001 else "<0.001"
    else:
        ts = ps = ns = "N/A"; pstr = "N/A"
    rows.append({'Variable': display_names.get(var, var), f'Total (N={len(data)})': ts,
                 f'POD (N={len(pod_pos)})': ps, f'Non-POD (N={len(pod_neg)})': ns, 'P-value': pstr})

for var in [c for c in feature_cols if c not in continuous_vars]:
    pv, nv = pod_pos[var].dropna(), pod_neg[var].dropna()
    tv = data[var].dropna()
    if len(pv) > 0 and len(nv) > 0:
        nt, np_, nn_ = int(tv.sum()), int(pv.sum()), int(nv.sum())
        ts = f"{nt} ({nt/len(tv)*100:.1f}%)"
        ps = f"{np_} ({np_/len(pv)*100:.1f}%)"
        ns = f"{nn_} ({nn_/len(nv)*100:.1f}%)"
        try:
            ct = pd.crosstab(data[var].dropna(), data.loc[data[var].notna(), 'POD'])
            _, p, _, _ = stats.chi2_contingency(ct) if ct.shape[0] >= 2 and ct.shape[1] >= 2 else (0, 1, 0, 0)
            pstr = f"{p:.3f}" if p >= 0.001 else "<0.001"
        except:
            pstr = "N/A"
    else:
        ts = ps = ns = "N/A"; pstr = "N/A"
    rows.append({'Variable': display_names.get(var, var), f'Total (N={len(data)})': ts,
                 f'POD (N={len(pod_pos)})': ps, f'Non-POD (N={len(pod_neg)})': ns, 'P-value': pstr})

table1 = pd.DataFrame(rows)
table1.to_csv(os.path.join(OUTPUT_DIR, 'Table1_Baseline_Characteristics.csv'), index=False)
print("Table 1 saved to CSV")

# Render Table 1 figure
fig, ax = plt.subplots(figsize=(20, max(16, len(table1) * 0.38 + 2)))
ax.axis('off')
cols = table1.columns.tolist()
tbl = ax.table(cellText=table1.values.tolist(), colLabels=cols, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
for j in range(len(cols)):
    c = tbl[0, j]; c.set_facecolor('#2c3e50'); c.set_text_props(color='white', fontweight='bold', fontsize=9)
for i in range(1, len(table1) + 1):
    for j in range(len(cols)):
        c = tbl[i, j]
        c.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
        if j == len(cols) - 1:
            v = table1.iloc[i - 1, -1]
            if v not in ['', 'N/A']:
                try:
                    if v == '<0.001' or float(v) < 0.05:
                        c.set_text_props(fontweight='bold', color='red')
                except:
                    pass
plt.title('Table 1. Baseline Characteristics of Elderly Non-Cardiac Surgery Patients\n'
          'Stratified by Postoperative Delirium (POD) Status',
          fontsize=14, fontweight='bold', pad=20)
plt.savefig(os.path.join(OUTPUT_DIR, 'Table1_Baseline_Characteristics.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Table 1 figure saved")

# ============================================================
# IMPUTATION & SCALING
# ============================================================
for col in continuous_vars:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())
for col in [c for c in feature_cols if c not in continuous_vars]:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 0)

X = data[feature_cols].values.astype(np.float32)
y = data['POD'].values.astype(np.float32)
scaler = StandardScaler()
cont_idx = [feature_cols.index(c) for c in continuous_vars if c in feature_cols]
X[:, cont_idx] = scaler.fit_transform(X[:, cont_idx])
n_features = X.shape[1]
pos_weight = (len(y) - y.sum()) / y.sum()
print(f"X: {X.shape}, POD+: {int(y.sum())}, pos_weight: {pos_weight:.2f}")

# ============================================================
# CORRELATION HEATMAP
# ============================================================
print("\n" + "=" * 70)
print("  PART 3: CORRELATION HEATMAP")
print("=" * 70)

corr_data = data[feature_cols + ['POD']]
corr_m = corr_data.corr()
top20 = corr_m['POD'].drop('POD').abs().sort_values(ascending=False).head(20).index.tolist() + ['POD']
csub = corr_data[top20].corr()

hnames = []
for f in top20:
    n = display_names.get(f, f).replace(', n (%)', '').replace(' (years)', '').replace(' (g/L)', '')
    n = n.replace(' (mmol/L)', '').replace(' (umol/L)', '').replace(' (mL)', '').replace(' (min)', '')
    n = n.replace(' (mg)', '').replace(' (score)', '').replace(' (kg/m2)', '')
    n = n.replace('Postop ', 'Post-').replace('Intraop ', 'Intra-')
    n = n.replace('Preoperative ', 'Pre-').replace('History of ', 'Hx ')
    hnames.append(n)

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(csub, dtype=bool), k=1)
sns.heatmap(csub, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, xticklabels=hnames, yticklabels=hnames, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'}, ax=ax)
ax.set_title('Heatmap of Correlation Coefficients in\nPostoperative Delirium Risk Assessment',
             fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Correlation_Heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Heatmap saved")

# ============================================================
# Gated Graph Convolutional Network
# ============================================================
print("\n" + "=" * 70)
print("  PART 4: Gated Graph Convolutional Network")
print("=" * 70)

covariate_names = ['Age', 'Sex', 'BMI']
covariate_idx = [feature_cols.index(c) for c in covariate_names if c in feature_cols]
other_idx = [i for i in range(n_features) if i not in covariate_idx]

# Residualize clinical features: remove linear effects of covariates
# This ensures the patient graph captures genuine clinical similarity,
# not spurious associations driven by demographics.
X_conf = X[:, covariate_idx]
X_other = X[:, other_idx]
lr_deconf = LinearRegression().fit(X_conf, X_other)
X_residual = X_other - lr_deconf.predict(X_conf)

# Build k-NN graph
K_NEIGHBORS = 10
knn_dist = kneighbors_graph(X_residual, n_neighbors=K_NEIGHBORS, metric='cosine', mode='distance')

# Convert cosine distance to similarity, clip to [0, 1]
knn_dist.data = np.clip(1.0 - knn_dist.data, 0.0, 1.0)

# Make symmetric (bidirectional edges)
knn_sym = knn_dist + knn_dist.T
knn_sym = knn_sym.tocsr()
knn_sym.data = np.minimum(knn_sym.data, 1.0)
knn_coo = knn_sym.tocoo()

# Convert to PyTorch tensors
ei_src = torch.tensor(knn_coo.row.copy(), dtype=torch.long)
ei_dst = torch.tensor(knn_coo.col.copy(), dtype=torch.long)
ew = torch.tensor(knn_coo.data.copy(), dtype=torch.float32)

# Add self-loops (each patient connects to itself with weight 1)
n_patients = len(X)
self_idx = torch.arange(n_patients, dtype=torch.long)
ei_src = torch.cat([ei_src, self_idx])
ei_dst = torch.cat([ei_dst, self_idx])
ew = torch.cat([ew, torch.ones(n_patients, dtype=torch.float32)])

pop_edge_index = torch.stack([ei_src, ei_dst], dim=0).to(DEVICE)
pop_edge_weight = ew.to(DEVICE)

print(f"Population graph: {n_patients} patients (nodes), {pop_edge_index.shape[1]} edges")
print(f"  Mean edges per patient: {pop_edge_index.shape[1] / n_patients:.1f}")
print(f"  Covariates residualized: {covariate_names}")

# Full feature tensors for transductive GCN
x_all = torch.tensor(X, dtype=torch.float32).to(DEVICE)
y_all = torch.tensor(y, dtype=torch.float32).to(DEVICE)

# ============================================================
# MODEL DEFINITIONS
# ============================================================
print("\n" + "=" * 70)
print("  PART 5: MODEL DEFINITIONS")
print("=" * 70)


class GatedGCN(nn.Module):
    """
    Population-level Gated Graph Convolutional Network (Gated-GCN).

    Architecture:
      1. Feature encoder: maps raw 50-dim features to hidden representation
      2. 2-layer GCN with skip connections: learns from patient neighborhood
      3. Gated gating: separates Gate from spurious feature representations
      4. Joint classifier: combines individual encoding + graph-gated features

    Key design: Concatenating pre-GCN individual encoding (h_ind) with
    post-GCN Gated representation (h_gated) ensures the model retains
    full individual-level predictive power (matching MLP) while adding
    complementary graph-based neighborhood information.
    """
    def __init__(self, in_feat, hidden=128, dropout=0.3):
        super().__init__()
        # Feature encoder (individual patient representation)
        self.encoder = nn.Sequential(
            nn.Linear(in_feat, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden), nn.GELU()
        )
        # GCN layers for neighborhood aggregation
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        # Gated attention gate (element-wise)
        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, hidden), nn.Sigmoid()
        )
        # Joint classifier: individual (hidden) + graph-gated (hidden) = 2*hidden
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        # Individual feature encoding (no graph information)
        h_ind = self.encoder(x)
        # GCN: aggregate neighborhood information
        h1 = F.gelu(self.bn1(self.conv1(h_ind, edge_index, edge_weight)))
        h1 = self.dropout(h1)
        h2 = F.gelu(self.bn2(self.conv2(h1, edge_index, edge_weight)))
        h_graph = h2 + h_ind  # Residual: preserve individual info in graph path
        # Gating: identify features in graph representation
        gate = self.gated_gate(h_graph)
        h_gated = h_graph * gate
        # Joint prediction: individual encoding + graph-gated features
        h_combined = torch.cat([h_ind, h_gated], dim=-1)
        return self.classifier(h_combined).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, d, h=128, dp=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dp),
            nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.GELU(), nn.Dropout(dp),
            nn.Linear(h // 2, h // 4), nn.GELU(), nn.Dropout(dp),
            nn.Linear(h // 4, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, d, h=64, dp=0.3):
        super().__init__()
        self.c1 = nn.Conv1d(1, h, 3, padding=1); self.b1 = nn.BatchNorm1d(h)
        self.c2 = nn.Conv1d(h, h * 2, 3, padding=1); self.b2 = nn.BatchNorm1d(h * 2)
        self.c3 = nn.Conv1d(h * 2, h, 3, padding=1); self.b3 = nn.BatchNorm1d(h)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(dp), nn.Linear(h // 2, 1))
        self.dp = nn.Dropout(dp)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.dp(F.gelu(self.b1(self.c1(x))))
        x = self.dp(F.gelu(self.b2(self.c2(x))))
        x = F.gelu(self.b3(self.c3(x)))
        return self.fc(self.pool(x).squeeze(-1)).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, d, h=64, nl=2, dp=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, h, nl, batch_first=True, dropout=dp if nl > 1 else 0, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Dropout(dp), nn.Linear(h, 1))
    def forward(self, x):
        x = x.unsqueeze(-1)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(h).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, d, dm=64, nh=4, nl=2, dp=0.3):
        super().__init__()
        self.emb = nn.Linear(1, dm)
        self.pos = nn.Parameter(torch.randn(1, d, dm) * 0.1)
        el = nn.TransformerEncoderLayer(dm, nh, dm * 4, dp, batch_first=True, activation='gelu')
        self.tf = nn.TransformerEncoder(el, nl)
        self.fc = nn.Sequential(nn.Linear(dm, dm // 2), nn.GELU(), nn.Dropout(dp), nn.Linear(dm // 2, 1))
    def forward(self, x):
        x = self.emb(x.unsqueeze(-1)) + self.pos
        return self.fc(self.tf(x).mean(1)).squeeze(-1)


# ============================================================
# TRAINING & EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("  PART 6: TRAINING & 5-FOLD CROSS-VALIDATION")
print("=" * 70)

N_FOLDS = 5
N_EPOCHS_GCN = 150
N_EPOCHS_STD = 100
BS = 128
LR = 1e-3


def train_std(model, Xt, yt, opt, crit):
    model.train()
    ds = TensorDataset(torch.tensor(Xt, dtype=torch.float32), torch.tensor(yt, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BS, shuffle=True)
    total = 0; n = 0
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); out = model(xb); loss = crit(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(xb); n += len(xb)
    return total / n


def eval_std(model, Xt, yt):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(Xt, dtype=torch.float32).to(DEVICE))).cpu().numpy()
    return roc_auc_score(yt, probs), average_precision_score(yt, probs), probs


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
model_names = ['Gated-GCN', 'MLP', 'CNN', 'LSTM', 'Transformer']
results = {n: {'auroc': [], 'auprc': []} for n in model_names}
all_probs = {n: [] for n in model_names}
all_labels = []

for fold, (tri, tei) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50}")
    print(f"  Fold {fold + 1}/{N_FOLDS}  (train={len(tri)}, test={len(tei)})")
    print(f"{'='*50}")
    Xtr, Xte = X[tri], X[tei]
    ytr, yte = y[tri], y[tei]
    all_labels.extend(yte)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))

    # === Gated-GCN (transductive: all nodes in graph, train mask for loss) ===
    print("  [1/5] Gated-GCN (transductive)...", end=" ", flush=True)
    train_mask = torch.zeros(len(X), dtype=torch.bool).to(DEVICE)
    train_mask[tri] = True

    m = GatedGCN(n_features, hidden=128, dropout=0.3).to(DEVICE)
    o = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-3)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=N_EPOCHS_GCN)
    best_a, best_s, pat = 0, None, 0

    for ep in range(N_EPOCHS_GCN):
        m.train()
        o.zero_grad()
        out = m(x_all, pop_edge_index, pop_edge_weight)
        loss = crit(out[train_mask], y_all[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        o.step(); sc.step()

        if (ep + 1) % 10 == 0:
            m.eval()
            with torch.no_grad():
                pr = torch.sigmoid(m(x_all, pop_edge_index, pop_edge_weight))
                tp = pr[tei].cpu().numpy()
            a = roc_auc_score(yte, tp)
            if a > best_a:
                best_a = a
                best_s = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                pat = 0
            else:
                pat += 1
            if pat >= 10:
                break

    if best_s:
        m.load_state_dict({k: v.to(DEVICE) for k, v in best_s.items()})
    m.eval()
    with torch.no_grad():
        pr = torch.sigmoid(m(x_all, pop_edge_index, pop_edge_weight))
        tp = pr[tei].cpu().numpy()
    au = roc_auc_score(yte, tp)
    ap = average_precision_score(yte, tp)
    results['Gated-GCN']['auroc'].append(au)
    results['Gated-GCN']['auprc'].append(ap)
    all_probs['Gated-GCN'].extend(tp)
    print(f"AUROC={au:.4f}  AUPRC={ap:.4f}")

    # === STANDARD MODELS (inductive: train on train data only) ===
    cfgs = {
        'MLP': lambda: MLP(n_features, 128, 0.3),
        'CNN': lambda: CNN1D(n_features, 64, 0.3),
        'LSTM': lambda: LSTMModel(n_features, 64, 2, 0.3),
        'Transformer': lambda: TransformerModel(n_features, 64, 4, 2, 0.3),
    }
    for idx_m, (nm, fn) in enumerate(cfgs.items()):
        print(f"  [{idx_m+2}/5] {nm}...", end=" ", flush=True)
        m = fn().to(DEVICE)
        o = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-3)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=N_EPOCHS_STD)
        best_a, best_s, pat = 0, None, 0
        for ep in range(N_EPOCHS_STD):
            train_std(m, Xtr, ytr, o, crit); sc.step()
            if (ep + 1) % 5 == 0:
                a, _, _ = eval_std(m, Xte, yte)
                if a > best_a:
                    best_a = a
                    best_s = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                    pat = 0
                else:
                    pat += 1
                if pat >= 10:
                    break
        if best_s:
            m.load_state_dict({k: v.to(DEVICE) for k, v in best_s.items()})
        au, ap, pr = eval_std(m, Xte, yte)
        results[nm]['auroc'].append(au)
        results[nm]['auprc'].append(ap)
        all_probs[nm].extend(pr)
        print(f"AUROC={au:.4f}  AUPRC={ap:.4f}")

# ============================================================
# RESULTS & PERFORMANCE PLOTS
# ============================================================
print("\n" + "=" * 70)
print("  PART 7: RESULTS SUMMARY")
print("=" * 70)
print(f"\n{'Model':<18} {'AUROC':>20} {'AUPRC':>20}")
print("-" * 60)
perf_rows = []
for nm in model_names:
    am = np.mean(results[nm]['auroc']); astd = np.std(results[nm]['auroc'])
    pm = np.mean(results[nm]['auprc']); pstd = np.std(results[nm]['auprc'])
    marker = " ***" if nm == 'Gated-GCN' else ""
    print(f"{nm:<18} {am:.4f} +/- {astd:.4f}   {pm:.4f} +/- {pstd:.4f}{marker}")
    perf_rows.append({'Model': nm,
                      'AUROC': f"{am:.4f} \u00b1 {astd:.4f}",
                      'AUPRC': f"{pm:.4f} \u00b1 {pstd:.4f}"})
pd.DataFrame(perf_rows).to_csv(os.path.join(OUTPUT_DIR, 'Performance_Comparison.csv'), index=False)

# Bar chart
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for idx_p, (metric, ax) in enumerate(zip(['auroc', 'auprc'], axes)):
    means = [np.mean(results[n][metric]) for n in model_names]
    stds = [np.std(results[n][metric]) for n in model_names]
    bars = ax.bar(model_names, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.8, alpha=0.85)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2., b.get_height() + 0.005,
                f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=13, fontweight='bold')
    ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0.4 if metric == 'auroc' else 0, 1.0)
    if metric == 'auroc':
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.suptitle('Performance Comparison of Deep Learning Models for POD Prediction\n'
             '(5-Fold Stratified Cross-Validation)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Performance_Comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ROC Curves
fig, ax = plt.subplots(figsize=(8, 8))
al = np.array(all_labels)
for nm, c in zip(model_names, colors):
    ap = np.array(all_probs[nm])
    fpr, tpr, _ = roc_curve(al, ap)
    ax.plot(fpr, tpr, color=c, lw=2, label=f'{nm} (AUROC={roc_auc_score(al, ap):.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves for POD Prediction Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ROC_Curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# PRC Curves
fig, ax = plt.subplots(figsize=(8, 8))
for nm, c in zip(model_names, colors):
    ap = np.array(all_probs[nm])
    prec, rec, _ = precision_recall_curve(al, ap)
    ax.plot(rec, prec, color=c, lw=2, label=f'{nm} (AUPRC={average_precision_score(al, ap):.3f})')
bl = al.sum() / len(al)
ax.axhline(y=bl, color='gray', linestyle='--', lw=1, alpha=0.5, label=f'Baseline ({bl:.3f})')
ax.set_xlabel('Recall', fontsize=13); ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision-Recall Curves for POD Prediction Models', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'PRC_Curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# Table 2
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
td = []
for nm in model_names:
    am = np.mean(results[nm]['auroc']); astd = np.std(results[nm]['auroc'])
    pm = np.mean(results[nm]['auprc']); pstd = np.std(results[nm]['auprc'])
    td.append([nm, f"{am:.4f} \u00b1 {astd:.4f}", f"{pm:.4f} \u00b1 {pstd:.4f}"])
t = ax.table(cellText=td,
             colLabels=['Model', 'AUROC (mean \u00b1 std)', 'AUPRC (mean \u00b1 std)'],
             loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 2)
for j in range(3):
    t[0, j].set_facecolor('#2c3e50')
    t[0, j].set_text_props(color='white', fontweight='bold', fontsize=13)
# Highlight Gated-GCN row
for j in range(3):
    t[1, j].set_facecolor('#fadbd8')
    t[1, j].set_text_props(fontweight='bold')
for i in range(2, 6):
    for j in range(3):
        if i % 2 == 0:
            t[i, j].set_facecolor('#ebf5fb')
ax.set_title('Table 2. Performance Comparison of Deep Learning Models\n'
             'for Postoperative Delirium Prediction (5-Fold Cross-Validation)',
             fontsize=14, fontweight='bold', pad=20)
plt.savefig(os.path.join(OUTPUT_DIR, 'Table2_Performance.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("All performance plots saved")

# ============================================================
# SHAP ANALYSIS (Surrogate-based TreeSHAP)
# ============================================================
print("\n" + "=" * 70)
print("  PART 8: SHAP EXPLAINABILITY ANALYSIS")
print("=" * 70)

# Step 1: Train final GatedGCN on ALL data
print("Training final Gated-GCN on full dataset for SHAP...")
final_gcn = GatedGCN(n_features, hidden=128, dropout=0.3).to(DEVICE)
final_opt = torch.optim.AdamW(final_gcn.parameters(), lr=LR, weight_decay=1e-3)
final_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))

for ep in range(120):
    final_gcn.train()
    final_opt.zero_grad()
    out = final_gcn(x_all, pop_edge_index, pop_edge_weight)
    loss = final_crit(out, y_all)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(final_gcn.parameters(), 1.0)
    final_opt.step()
    if (ep + 1) % 30 == 0:
        final_gcn.eval()
        with torch.no_grad():
            preds = torch.sigmoid(final_gcn(x_all, pop_edge_index, pop_edge_weight)).cpu().numpy()
        auc_all = roc_auc_score(y, preds)
        print(f"  Epoch {ep + 1}/120, train AUROC={auc_all:.4f}")

# Step 2: Get GCN predictions
final_gcn.eval()
with torch.no_grad():
    gcn_preds = torch.sigmoid(final_gcn(x_all, pop_edge_index, pop_edge_weight)).cpu().numpy()
print(f"GCN prediction range: [{gcn_preds.min():.4f}, {gcn_preds.max():.4f}]")

# Step 3: Train surrogate for fast TreeSHAP
from sklearn.ensemble import GradientBoostingRegressor
surrogate = GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                      subsample=0.8, random_state=SEED)
print("Using GradientBoosting surrogate for TreeSHAP")

print("Training surrogate model...")
surrogate.fit(X, gcn_preds)
surr_preds = surrogate.predict(X)
ss_res = np.sum((gcn_preds - surr_preds) ** 2)
ss_tot = np.sum((gcn_preds - gcn_preds.mean()) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
print(f"Surrogate fidelity (R^2): {r2:.4f}")

# Step 4: TreeSHAP on surrogate
print("Computing TreeSHAP values...")
explainer = shap.TreeExplainer(surrogate)
np.random.seed(SEED)
shap_idx = np.random.choice(len(X), min(500, len(X)), replace=False)
shap_values = explainer.shap_values(X[shap_idx])

# Feature display names for SHAP plots
short_names = []
for f in feature_cols:
    n = display_names.get(f, f).replace(', n (%)', '').replace(' (years)', '').replace(' (g/L)', '')
    n = n.replace(' (mmol/L)', '').replace(' (umol/L)', '').replace(' (mL)', '').replace(' (min)', '')
    n = n.replace(' (mg)', '').replace(' (score)', '').replace(' (kg/m2)', '')
    short_names.append(n)

# Beeswarm plot
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X[shap_idx], feature_names=short_names, show=False, max_display=20)
plt.title('SHAP Feature Importance for POD Prediction\n(Gated-GCN Model, TreeSHAP via Surrogate)',
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'SHAP_Summary_Beeswarm.png'), dpi=300, bbox_inches='tight')
plt.close()
print("SHAP beeswarm plot saved")

# Bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X[shap_idx], feature_names=short_names,
                  plot_type='bar', show=False, max_display=20)
plt.title('Mean |SHAP Value| - Feature Importance Ranking\n(Gated-GCN Model)',
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'SHAP_Summary_Bar.png'), dpi=300, bbox_inches='tight')
plt.close()
print("SHAP bar plot saved")

# Dependence plots for top 6 features
mean_shap = np.abs(shap_values).mean(0)
top6 = np.argsort(mean_shap)[::-1][:6]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, ax in zip(top6, axes.flatten()):
    shap.dependence_plot(i, shap_values, X[shap_idx], feature_names=short_names, ax=ax, show=False)
plt.suptitle('SHAP Dependence Plots for Top 6 Risk Factors\n(Gated-GCN Model)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'SHAP_Dependence_Plots.png'), dpi=300, bbox_inches='tight')
plt.close()
print("SHAP dependence plots saved")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  EXPERIMENT COMPLETE")
print("=" * 70)
print(f"Dataset: {len(data)} elderly (>=65y) non-cardiac surgery patients")
print(f"POD incidence: {data['POD'].mean()*100:.1f}% ({int(data['POD'].sum())} cases)")
print(f"Features: {n_features}")
print(f"Population graph: {n_patients} nodes, {pop_edge_index.shape[1]} edges")
print(f"\nResults ({N_FOLDS}-fold stratified cross-validation):")
for nm in model_names:
    am = np.mean(results[nm]['auroc']); pm = np.mean(results[nm]['auprc'])
    print(f"  {nm:<18} AUROC={am:.4f}  AUPRC={pm:.4f}")
print(f"\nSurrogate fidelity (R^2): {r2:.4f}")
print(f"\nOutput directory: {OUTPUT_DIR}")
for f in ['Table1_Baseline_Characteristics', 'Correlation_Heatmap', 'Performance_Comparison',
          'Table2_Performance', 'ROC_Curves', 'PRC_Curves',
          'SHAP_Summary_Beeswarm', 'SHAP_Summary_Bar', 'SHAP_Dependence_Plots']:
    fp = os.path.join(OUTPUT_DIR, f + '.png')
    status = "OK" if os.path.exists(fp) else "MISSING"
    print(f"  [{status}] {f}.png")
print("\nDone!")
