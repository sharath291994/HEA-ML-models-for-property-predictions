"""HEA melting-point regression: a synthetic-data learning exercise.

Install dependencies: pip install numpy pandas matplotlib scikit-learn
Put this script and AlCoCrFeNi_synthetic_melting_1000.csv in the same folder.
Run in an IDE or execute this file to train, evaluate, and plot.

Only melting point is a target. Hardness and tensile strength are NOT inputs.
Synthetic targets are called 'experimental' only for the requested plots.
Alloy formation energy is NOT an elemental property. No DFT/CALPHAD energies
are available here, so an explicitly invented pair-interaction proxy is used.
Replace that proxy with independently obtained, composition-matched energies
before scientific use. Pure elements have zero formation energy relative to
their elemental reference states; averaging those zeros is not useful.

Elemental constants are rounded reference values, not alloy predictions.
Sources: https://periodic-table.rsc.org (element pages 13, 24, 26, 27, 28).
Radius convention: covalent radius, consistently used for all five elements.
The size mismatch below is therefore a covalent-radius descriptor, not a
claim about measured metallic radii or the actual crystal structure.

All descriptors are functions of composition. Correlated descriptors can
share importance; importance is predictive, not a causal interpretation.
The weighted elemental melting point is a legitimate descriptor, but this
synthetic target was built from it, making this exercise unusually easy.
An alloy generally has a solidus/liquidus interval; this toy target is one
scalar temperature and does not distinguish those temperatures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
})


class HEAMeltingPointPredictor:
    """Load alloy compositions, train regressors, evaluate, and plot results."""

    def __init__(self):
        self.random_state = 42
        self.test_fraction = 0.20
        self.cv_folds = 5
        self.save_plots = True
        self.base_dir = Path(__file__).resolve().parent
        self.csv_path = self.base_dir / 'AlCoCrFeNi_synthetic_melting_1000.csv'
        self.elements = ['Al', 'Co', 'Cr', 'Fe', 'Ni']
        self.composition_columns = [
            f'{element}_at_pct' for element in self.elements
        ]
        self.target = 'synthetic_melting_point_K'
        self.figures = []
        self.results = []
        self.cv_rmse = {}
        self.cv_rmse_summary = {}

    def load_and_validate_data(self):
        """Read the input CSV and validate required data."""
        self.data = pd.read_csv(self.csv_path)
        required = self.composition_columns + [self.target]
        if not set(required).issubset(self.data.columns):
            raise ValueError(f'CSV must contain these columns: {required}')

        values = self.data[required].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Composition and target columns must be finite numbers.')

        atomic_percent = self.data[self.composition_columns].to_numpy(dtype=float)
        if (atomic_percent <= 0).any():
            raise ValueError('This exercise expects all five element fractions > 0.')
        if not np.allclose(atomic_percent.sum(axis=1), 100.0, atol=0.001):
            raise ValueError('Each composition must sum to 100 atomic %.')
        if self.data.duplicated(subset=self.composition_columns).any():
            raise ValueError('Duplicate compositions could leak across train/test sets.')

        self.compositions = atomic_percent / 100.0
        self.y = self.data[self.target].astype(float)

    def build_descriptors(self):
        """Calculate composition-based model features."""
        # All arrays follow elements order: Al, Co, Cr, Fe, Ni.
        atomic_mass = np.array([26.9815, 58.9332, 51.9961, 55.845, 58.6934])
        covalent_radius = np.array([1.24, 1.18, 1.30, 1.24, 1.17])
        element_melting_K = np.array([933.47, 1768.0, 2180.0, 1811.0, 1728.0])
        electronegativity = np.array([1.61, 1.88, 1.66, 1.83, 1.91])
        valence_electrons = np.array([3.0, 9.0, 6.0, 8.0, 10.0])
        c = self.compositions

        self.X = pd.DataFrame(index=self.data.index)
        # Ni is implied by the sum-to-one constraint. Including all five
        # fractions would make them linearly dependent with the intercept.
        for i, element in enumerate(self.elements[:-1]):
            self.X[f'{element}_atomic_fraction'] = c[:, i]

        self.X['mean_atomic_mass_u'] = c @ atomic_mass
        self.X['mean_covalent_radius_A'] = c @ covalent_radius
        self.X['mean_element_melting_K'] = c @ element_melting_K
        self.X['mean_electronegativity'] = c @ electronegativity
        self.X['valence_electron_concentration'] = c @ valence_electrons

        mean_radius = self.X['mean_covalent_radius_A'].to_numpy()
        mean_mass = self.X['mean_atomic_mass_u'].to_numpy()
        mean_en = self.X['mean_electronegativity'].to_numpy()
        mean_tm = self.X['mean_element_melting_K'].to_numpy()

        self.X['covalent_size_mismatch_pct'] = 100 * np.sqrt(
            np.sum(c * (1 - covalent_radius / mean_radius[:, None]) ** 2, axis=1)
        )
        self.X['atomic_mass_std_u'] = np.sqrt(
            np.sum(c * (atomic_mass - mean_mass[:, None]) ** 2, axis=1)
        )
        self.X['electronegativity_std'] = np.sqrt(
            np.sum(c * (electronegativity - mean_en[:, None]) ** 2, axis=1)
        )
        self.X['element_melting_std_K'] = np.sqrt(
            np.sum(c * (element_melting_K - mean_tm[:, None]) ** 2, axis=1)
        )
        self.X['ideal_config_entropy_J_mol_K'] = (
            -8.314462618 * np.sum(c * np.log(c), axis=1)
        )

        # E_proxy sums 4*c_i*c_j*pair_energy_ij for i<j. These illustrative
        # coefficients were not fitted and are not measured or DFT-derived.
        pair_energy = np.array([
            [0.00, -0.18, -0.08, -0.12, -0.22],
            [-0.18, 0.00, -0.02, -0.01, -0.01],
            [-0.08, -0.02, 0.00, -0.03, -0.04],
            [-0.12, -0.01, -0.03, 0.00, -0.02],
            [-0.22, -0.01, -0.04, -0.02, 0.00],
        ])
        formation_energy_proxy = np.zeros(len(self.data))
        for i in range(len(self.elements)):
            for j in range(i + 1, len(self.elements)):
                formation_energy_proxy += (
                    4 * c[:, i] * c[:, j] * pair_energy[i, j]
                )
        self.X['synthetic_formation_energy_proxy_eV_atom'] = formation_energy_proxy

        if not np.isfinite(self.X.to_numpy()).all():
            raise ValueError('Generated model descriptors contain non-finite values.')

    def split_data_and_define_models(self):
        """Create training/test partitions and the regression models."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_fraction,
            random_state=self.random_state
        )
        self.models = {
            'Linear Regression': make_pipeline(
                StandardScaler(), LinearRegression()
            ),
            'Decision Tree': DecisionTreeRegressor(
                max_depth=8, min_samples_leaf=5,
                random_state=self.random_state
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, max_features=1.0,
                random_state=self.random_state, n_jobs=-1
            ),
        }
        # 5-fold cross-validation means the training data is split into 5
        # roughly equal parts. On each pass, 4 parts train and 1 part validates;
        # each row is used in validation exactly once. This is not "remove 5
        # random rows"; it is a repeated out-of-fold evaluation over all rows.
        self.cv = KFold(
            n_splits=self.cv_folds, shuffle=True,
            random_state=self.random_state
        )
        self.scoring = {
            # Use the negative RMSE so scikit-learn's "higher is better"
            # convention matches the regression metric where lower RMSE is better.
            'rmse': 'neg_root_mean_squared_error',
        }

    def create_prediction_plot(self, name, test_prediction, test_rmse, test_r2):
        """Create and retain a held-out prediction plot."""
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(self.y_test, test_prediction, alpha=0.65, s=30, edgecolor='none')
        lower = min(self.y_test.min(), test_prediction.min()) - 20
        upper = max(self.y_test.max(), test_prediction.max()) + 20
        ax.plot([lower, upper], [lower, upper], 'k--', label='Perfect prediction')
        ax.set(
            xlim=(lower, upper),
            ylim=(lower, upper),
            xlabel='CALPHAD melting point prediction, K',
            ylabel='Predicted melting point, K',
            title=f'{name}: held-out test predictions'
        )
        if name == 'Linear Regression':
            ax.xaxis.label.set_size(18)
            ax.yaxis.label.set_size(18)
            ax.tick_params(axis='both', labelsize=18)
        ax.set_aspect('equal', adjustable='box')
        ax.text(
            0.04, 0.96,
            f'Test RMSE = {test_rmse:.2f} K\nTest R² = {test_r2:.3f}',
            transform=ax.transAxes, va='top',
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'grey'}
        )
        ax.legend(loc='lower right')
        ax.grid(alpha=0.2)
        fig.tight_layout()
        self.figures.append((
            name.lower().replace(' ', '_') + '_predictions', fig
        ))

    def create_feature_importance_plot(self, name, model):
        """Create and retain a permutation feature-importance plot."""
        # Positive values mean shuffling the feature increased test RMSE.
        # Treat this as a post-evaluation diagnostic, not a tuning signal.
        importance = permutation_importance(
            model, self.X_test, self.y_test,
            scoring='neg_root_mean_squared_error', n_repeats=15,
            random_state=self.random_state, n_jobs=1
        )
        order = np.argsort(importance.importances_mean)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            self.X.columns[order], importance.importances_mean[order],
            xerr=importance.importances_std[order], alpha=0.8, capsize=3
        )
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set(
            xlabel='Increase in test RMSE when shuffled (K)',
            title=f'{name}: permutation feature importance'
        )
        font_size = 18 if name == 'Linear Regression' else 20
        ax.xaxis.label.set_size(font_size)
        ax.yaxis.label.set_size(font_size)
        ax.tick_params(axis='both', labelsize=font_size)
        ax.grid(axis='x', alpha=0.2)
        fig.tight_layout()
        self.figures.append((
            name.lower().replace(' ', '_') + '_feature_importance', fig
        ))

    def train_and_evaluate_models(self):
        """Cross-validate, fit, evaluate, and create per-model plots."""
        for name, model in self.models.items():
            # The scaler is fitted independently inside each CV training fold.
            scores = cross_validate(
                model, self.X_train, self.y_train,
                cv=self.cv, scoring=self.scoring
            )
            self.cv_rmse[name] = -scores['test_rmse']
            self.cv_rmse_summary[name] = self.cv_rmse[name].mean()

            model.fit(self.X_train, self.y_train)
            train_prediction = model.predict(self.X_train)
            test_prediction = model.predict(self.X_test)
            train_rmse = np.sqrt(
                mean_squared_error(self.y_train, train_prediction)
            )
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_prediction))
            test_r2 = r2_score(self.y_test, test_prediction)

            self.results.append({
                'Model': name,
                'Train RMSE (K)': train_rmse,
                'Test RMSE (K)': test_rmse,
                'Test R2': test_r2,
                'CV RMSE mean (K)': self.cv_rmse[name].mean(),
                'CV RMSE SD (K)': self.cv_rmse[name].std(ddof=1),
            })
            self.create_prediction_plot(
                name, test_prediction, test_rmse, test_r2
            )
            self.create_feature_importance_plot(name, model)

    def create_metric_plots(self):
        """Create the RMSE comparison and cross-validation metric plots."""
        self.summary = pd.DataFrame(self.results).set_index('Model')
        names = list(self.models)
        positions = np.arange(len(names))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(
            positions - 0.2, self.summary['Train RMSE (K)'],
            width=0.4, label='Training'
        )
        ax.bar(
            positions + 0.2, self.summary['Test RMSE (K)'],
            width=0.4, label='Held-out test'
        )
        ax.set_xticks(positions, names)
        ax.set(ylabel='RMSE (K)', title='Training and test RMSE')
        ax.legend(loc='upper left', fontsize=14)
        ax.grid(axis='y', alpha=0.2)
        fig.tight_layout()
        self.figures.append(('rmse_comparison', fig))

        # Explicit summary figure for the 5-fold CV score: each bar is the mean
        # RMSE across the 5 validation folds, so lower is better.
        fig, ax = plt.subplots(figsize=(12, 8))
        mean_cv_rmse = [self.cv_rmse_summary[name] for name in names]
        ax.bar(positions, mean_cv_rmse, alpha=0.75, color='steelblue')
        ax.set_xticks(positions, names)
        ax.set(
            ylabel='5-fold CV score (RMSE, K)',
            title='5-fold CV score'
        )
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        ax.title.set_size(16)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(axis='y', alpha=0.2)
        ax.text(
            0.0, -0.16,
            'Each bar = mean RMSE over the 5 validation folds. Lower RMSE is better.',
            transform=ax.transAxes, fontsize=16
        )
        fig.tight_layout()
        self.figures.append(('5_fold_cv_score', fig))

        fig, ax = plt.subplots(figsize=(12, 8))
        means = [self.cv_rmse[name].mean() for name in names]
        stds = [self.cv_rmse[name].std(ddof=1) for name in names]
        ax.bar(positions, means, yerr=stds, capsize=6, alpha=0.65)
        offsets = np.linspace(-0.12, 0.12, self.cv_folds)
        for i, name in enumerate(names):
            ax.scatter(
                i + offsets, self.cv_rmse[name],
                color='black', s=25, zorder=3
            )
        ax.set_xticks(positions, names)
        ax.set(
            ylabel='CV score (RMSE, K)',
            title=f'{self.cv_folds}-fold cross-validation scores'
        )
        ax.xaxis.label.set_size(26)
        ax.yaxis.label.set_size(26)
        ax.title.set_size(26)
        ax.tick_params(axis='both', labelsize=26)
        ax.grid(axis='y', alpha=0.2)
        fig.tight_layout()
        self.figures.append(('cv_scores', fig))

    def display_and_save_results(self):
        """Print metrics, save PNGs, and display the generated figures."""
        print('\nSynthetic-data exercise: no experimental or DFT claims.')
        print(
            f'Rows: {len(self.data)} | Training: {len(self.X_train)} '
            f'| Test: {len(self.X_test)}'
        )
        print('\n' + self.summary.round(3).to_string())
        print(
            '\nNine separate figures: 3 predictions, 3 importances, '
            'RMSE comparison, mean 5-fold CV score, fold-wise CV scores.'
        )
        if self.save_plots:
            output_dir = self.base_dir / 'HEA_ML_plots'
            output_dir.mkdir(parents=True, exist_ok=True)
            for filename, fig in self.figures:
                if filename == '5_fold_cv_score':
                    text_artists = []
                    for ax in fig.axes:
                        text_artists.extend([
                            ax.xaxis.label,
                            ax.yaxis.label,
                            ax.title,
                            *ax.get_xticklabels(),
                            *ax.get_yticklabels(),
                            *ax.texts,
                        ])
                    original_sizes = [
                        artist.get_fontsize() for artist in text_artists
                    ]
                    try:
                        for artist in text_artists:
                            artist.set_fontsize(14)
                        fig.savefig(
                            output_dir / f'{filename}.png',
                            dpi=180, bbox_inches='tight'
                        )
                    finally:
                        for artist, size in zip(text_artists, original_sizes):
                            artist.set_fontsize(size)
                else:
                    fig.savefig(
                        output_dir / f'{filename}.png',
                        dpi=180, bbox_inches='tight'
                    )
        plt.show()

    def run(self):
        """Execute the complete data-to-results workflow."""
        self.load_and_validate_data()
        self.build_descriptors()
        self.split_data_and_define_models()
        self.train_and_evaluate_models()
        self.create_metric_plots()
        self.display_and_save_results()


if __name__ == '__main__':
    predictor = HEAMeltingPointPredictor()
    predictor.run()
