from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV


def get_default_classifiers(probability=True, random_state=42, n_neighbors=None):
    if n_neighbors is None:
        n_neighbors = 4

    default_clfs = {
        "SVMR": SVC(kernel="rbf", gamma="auto", probability=probability, random_state=random_state),
        "SVML": SVC(kernel="linear", probability=probability, random_state=random_state),
        "SVMS": SVC(kernel="sigmoid", probability=probability, random_state=random_state),
        "RF": RandomForestClassifier(n_jobs=-1, random_state=random_state, n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1),
        "DCT": DecisionTreeClassifier(random_state=random_state),
        "LR": LogisticRegression(n_jobs=-1),
        "GBC": GradientBoostingClassifier(random_state=random_state),
        "ABC": AdaBoostClassifier(random_state=random_state),
        "ETC": ExtraTreesClassifier(n_jobs=-1, random_state=random_state),
        "MLP": MLPClassifier(random_state=random_state),
        "XGB": XGBClassifier(n_jobs=-1, random_state=random_state),
        "LGBM": LGBMClassifier(n_jobs=-1, random_state=random_state)
    }

    # Criar VotingClassifiers separadamente
    voting_hard = VotingClassifier(
        estimators=list(default_clfs.items()),
        voting="hard",
        n_jobs=-1
    )

    voting_soft = VotingClassifier(
        estimators=list(default_clfs.items()),
        voting="soft",
        n_jobs=-1
    )

    # Criar um GridSearchCV apenas para o SVM com kernel RBF
    svm_param_grid = {
        'C': [0.1, 1],
        'gamma': [1, 0.1],
    }
    
    grid_svm = GridSearchCV(
        SVC(probability=probability, random_state=random_state),
        param_grid=svm_param_grid,
        cv=3,
        n_jobs=-1
    )

    # Adicionar os classificadores ensemble ao dicionário
    return {
        **default_clfs,
        "HV": voting_hard,
        "SV": voting_soft,
        "SV-Grid": grid_svm,
    }

def set_classifier(model_name, clfs_dict):
    """Configura o classificador com base no nome fornecido."""
    clf = clfs_dict.get(model_name)
    if clf is None:
        raise ValueError(f"Modelo {model_name} não está disponível.")
    return clf
