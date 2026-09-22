# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
while not (PROJECT_ROOT / "modeling_pipeline.py").is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError("Run this code from the MMDLPC_Code_PDF folder.")
    PROJECT_ROOT = PROJECT_ROOT.parent
CODE_DIR = PROJECT_ROOT / "01_Clinical_and_Cohort/01_Cohort_and_Baseline"
DATA_ROOT = PROJECT_ROOT.parent / "MMDLPC_Code_PDF_Data"
OUTPUT_DIR = CODE_DIR
sys.path.insert(0, str(CODE_DIR))
INPUT_FILE = CODE_DIR / 'Figure1_clinical_genomic.xlsx'
# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse, Patch
import numpy as np
import pandas as pd

GENES = ['ATM', 'BARD1', 'BRCA1', 'BRCA2', 'BRIP1', 'CDK12', 'CHEK1', 'CHEK2',
         'FANCL', 'PALB2', 'RAD51B', 'RAD51C', 'RAD51D', 'RAD54L']
FIELDS = ['Pathology', 'Radiology', 'Age', 'PSA', 'pT', 'pN', 'pM', 'BCR', 'ResidualTumor', 'ISUP', 'HRD']
LABELS = ['Pathology', 'Radiology', 'Age', 'PSA', 'pT stage', 'pN stage', 'pM stage', 'BCR', 'Residual tumor', 'ISUP', 'HRD status']
PALETTES = {
    'Pathology': {'0': '#FFFFFF', '1': '#FBC999'},
    'Radiology': {'0': '#FFFFFF', '1': '#F5C9D0'},
    'Age': {'<=65': '#F0E7F4', '66-70': '#CEABCE', '71-75': '#B96DA7', '>75': '#CA185B'},
    'PSA': {'<4': '#ECD6A9', '4-20': '#BE551A', '>20': '#E89046'},
    'pT': {'pT2': '#8DC6DE', 'pT3': '#4594BA', 'pT4': '#217198', 'pTx': '#C7C7C7'},
    'pN': {'pN0': '#BBD760', 'pN1': '#78BD83', 'pNx': '#C7C7C7'},
    'pM': {'pM0': '#B18C14', 'pM1': '#946136', 'pMx': '#C7C7C7'},
    'BCR': {'NO': '#FFFFFF', 'YES': '#8D5B8B'},
    'ResidualTumor': {'NO': '#FFFFFF', 'YES': '#63376E'},
    'ISUP': {'1': '#F8C8C9', '2': '#EF999E', '3': '#DE5B6A', '4': '#BC2745', '5': '#941433'},
    'HRD': {'0': '#FFFFFF', '1': '#EF762C'},
}
MISSING = '#A9B1B7'


def load_input(path):
    if not path.is_file():
        raise FileNotFoundError(f'Provide the verified patient-level workbook: {path}. Required sheets: Patients and Alterations.')
    patients = pd.read_excel(path, sheet_name='Patients', dtype={'ID': str})
    alterations = pd.read_excel(path, sheet_name='Alterations', dtype={'ID': str})
    for frame, required, name in [(patients, ['ID', 'Cohort'] + FIELDS, 'Patients'),
                                   (alterations, ['ID'] + GENES, 'Alterations')]:
        absent = set(required) - set(frame)
        if absent:
            raise ValueError(f'{name}: missing columns {sorted(absent)}')
        if frame.ID.isna().any():
            raise ValueError(f'{name}: missing IDs')
        frame['ID'] = frame.ID.str.strip()
        if frame.ID.duplicated().any() or frame.ID.eq('').any():
            raise ValueError(f'{name}: duplicate or empty IDs')
    if set(patients.Cohort) != {'CPGEA', 'TCGA'}:
        raise ValueError('Cohort must contain CPGEA and TCGA')
    if set(patients.ID) != set(alterations.ID):
        raise ValueError('Patients and Alterations must contain exactly the same IDs; use NA for unassayed genes')
    for column in ['Pathology', 'Radiology']:
        if not patients[column].isin([0, 1]).all():
            raise ValueError(f'{column}: complete 0/1 modality-availability flags are required')
    if not patients.HRD.dropna().isin([0, 1]).all():
        raise ValueError('HRD must be 0/1 or NA')
    if not alterations[GENES].stack().isin([0, 1]).all():
        raise ValueError('Alterations must be 0=no alteration, 1=alteration, NA=unassayed')
    data = patients.merge(alterations, on='ID', validate='one_to_one')
    if 'PlotOrder' in data:
        if data.PlotOrder.isna().any() or data.duplicated(['Cohort', 'PlotOrder']).any():
            raise ValueError('PlotOrder must be complete and unique within each cohort')
    return data


def annotation_values(data):
    encoded = {}
    for field in FIELDS:
        values = data[field]
        if field in ['Age', 'PSA']:
            numeric = pd.to_numeric(values, errors='raise')
            if (numeric.dropna() < 0).any():
                raise ValueError(f'Negative {field}')
            if field == 'Age':
                values = pd.cut(numeric, [-np.inf, 65, 70, 75, np.inf], labels=['<=65', '66-70', '71-75', '>75']).astype(object)
            else:
                values = pd.Series(np.select([numeric < 4, numeric <= 20], ['<4', '4-20'], default='>20'), index=data.index)
                values[numeric.isna()] = np.nan
        elif field in ['Pathology', 'Radiology', 'ISUP', 'HRD']:
            numeric = pd.to_numeric(values, errors='raise')
            values = numeric.map(lambda x: str(int(x)) if pd.notna(x) and x == int(x) else (np.nan if pd.isna(x) else str(x)))
        else:
            values = values.map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
        unknown = set(values.dropna()) - set(PALETTES[field])
        if unknown:
            raise ValueError(f'{field}: unsupported categories {sorted(unknown)}')
        encoded[field] = values
    return encoded


def draw_cohort_overlap(ax, data):
    ax.set(xlim=(0, 1), ylim=(0, 2))
    ax.axis('off')
    ax.text(0, 2.04, 'B', fontsize=16, weight='bold')
    for cohort, base in [('CPGEA', 1.06), ('TCGA', .08)]:
        subset = data.loc[data.Cohort.eq(cohort)]
        ax.text(.5, base + .82, cohort, ha='center', fontsize=11)
        ax.add_patch(Ellipse((.5, base + .4), .98, .69, facecolor='#E4DBCA', edgecolor='#777777', lw=.8))
        ax.add_patch(Ellipse((.36, base + .3), .56, .39, facecolor='#AD78A0', edgecolor='#777777', alpha=.6, lw=.8))
        ax.add_patch(Ellipse((.66, base + .3), .56, .39, facecolor='#77A4BE', edgecolor='#777777', alpha=.6, lw=.8))
        ax.text(.5, base + .59, f'Clinicogenomic\n{len(subset)}', ha='center', va='center', fontsize=9)
        ax.text(.235, base + .28, f'Patho-\ngenomic\n{int(subset.Pathology.sum())}', ha='center', va='center', fontsize=9)
        ax.text(.78, base + .28, f'Radio-\ngenomic\n{int(subset.Radiology.sum())}', ha='center', va='center', fontsize=9)
        overlap = int((subset.Pathology.eq(1) & subset.Radiology.eq(1)).sum())
        ax.text(.51, base + .28, f'Trimodal\n{overlap}', ha='center', va='center', fontsize=9)


def draw_oncoprint(fig, slot, data, panel):
    counts = data[GENES].sum()
    genes = counts.sort_values(ascending=False, kind='stable').index.tolist()
    sort_cols = ['PlotOrder'] if 'PlotOrder' in data else ['HRD'] + genes + ['ID']
    ascending = True if 'PlotOrder' in data else [False] * (len(sort_cols) - 1) + [True]
    data = data.sort_values(sort_cols, ascending=ascending, na_position='last', kind='stable')
    encoded = annotation_values(data)
    colors = ['#FFFFFF', MISSING, '#CCCCCC', '#F36B21']
    codes = {color: i for i, color in enumerate(colors)}
    def color_code(color):
        if color not in codes:
            codes[color] = len(colors)
            colors.append(color)
        return codes[color]
    rows = []
    for field in FIELDS:
        rows.append([color_code(PALETTES[field].get(v, MISSING)) for v in encoded[field]])
    rows.append([0] * len(data))
    for gene in genes:
        rows.append([1 if pd.isna(v) else (3 if v == 1 else 2) for v in data[gene]])
    matrix = np.asarray(rows)
    grid = slot.subgridspec(1, 3, width_ratios=[20, 3.2, 2], wspace=.02)
    ax = fig.add_subplot(grid[0, 0])
    labels = fig.add_subplot(grid[0, 1], sharey=ax)
    bars = fig.add_subplot(grid[0, 2], sharey=ax)
    ax.pcolormesh(np.arange(len(data) + 1), np.arange(len(rows) + 1), matrix,
                  cmap=ListedColormap(colors), vmin=-.5, vmax=len(colors)-.5,
                  edgecolors='white', linewidth=.13, antialiased=False)
    ax.set(xlim=(0, len(data)), ylim=(len(rows), 0))
    ax.axis('off')
    ax.set_title(f'{panel}  {data.Cohort.iloc[0]} (n={len(data)})', loc='left', fontsize=12, weight='bold', pad=12)
    labels.set_xlim(0, 1)
    labels.axis('off')
    for i, label in enumerate(LABELS):
        labels.text(.02, i + .5, label, va='center', fontsize=8)
    for i, gene in enumerate(genes, start=len(FIELDS) + 1):
        count, denominator = int(data[gene].sum()), int(data[gene].notna().sum())
        percentage = f'{100 * count / denominator:.0f}%' if denominator else 'NA'
        ax.text(-.008, i + .5, percentage, transform=ax.get_yaxis_transform(), va='center', ha='right', fontsize=8)
        labels.text(.02, i + .5, gene, va='center', fontsize=8)
        bars.barh(i + .5, count, height=.77, color='#F36B21')
    bars.set_xlim(0, max(1, counts.max()) * 1.12)
    bars.spines[['top', 'left', 'right']].set_visible(False)
    bars.tick_params(axis='y', left=False, labelleft=False)
    bars.tick_params(axis='x', labelsize=7)
    bars.set_xlabel('Count', fontsize=7)
    bars.xaxis.set_major_locator(plt.MaxNLocator(3, integer=True))


def make_figure(data, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(14, 11.5))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.6, 1, 1, 1], height_ratios=[1, 1], hspace=.18, wspace=.22)
    draw_cohort_overlap(fig.add_subplot(grid[0, 0]), data)
    draw_oncoprint(fig, grid[0, 1:], data.loc[data.Cohort.eq('CPGEA')], 'C')
    draw_oncoprint(fig, grid[1, :], data.loc[data.Cohort.eq('TCGA')], 'D')
    handles = [Patch(facecolor=color, edgecolor='#BBBBBB', label=f'{field}: {category}')
               for field in FIELDS for category, color in PALETTES[field].items()]
    handles += [Patch(color='#F36B21', label='Alteration'), Patch(color='#CCCCCC', label='No alteration'), Patch(color=MISSING, label='Missing / unassayed')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .022), ncol=7, frameon=False, fontsize=7)
    fig.text(.055, .012, 'Percentages use assayed (nonmissing) patients per gene. Ellipses show membership; their areas are schematic.', fontsize=7)
    fig.subplots_adjust(left=.055, right=.97, top=.96, bottom=.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)

# %%
data = load_input(INPUT_FILE)
annotation_values(data)

# %%
make_figure(data, OUTPUT_DIR / 'Figure1B_D_Cohort_Oncoprint.pdf')
print(data.groupby('Cohort').agg(N=('ID', 'size'), Pathology=('Pathology', 'sum'), Radiology=('Radiology', 'sum')).to_string())
