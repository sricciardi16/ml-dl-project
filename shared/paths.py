import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing this file (shared folder)
current_dir = os.path.dirname(current_file_path)

# Get the parent directory (PROJECT_ROOT) - navigating up two levels from this file
PROJECT_ROOT = os.path.dirname(current_dir)

# Define base directory constants using os.path.join starting from PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
ANALYSES_DIR = os.path.join(DATA_DIR, '_analyses')
DATASETS_DIR = os.path.join(DATA_DIR, 'a_datasets')
MODELS_BASE_DIR = os.path.join(DATA_DIR, 'b_models')
MODELS_TRAINED_DIR = os.path.join(MODELS_BASE_DIR, 'trained')
MODELS_EVAL_DIR = os.path.join(MODELS_BASE_DIR, 'evaluations')
MODELS_LATENT_DIR = os.path.join(MODELS_BASE_DIR, 'latent_features')
ATTACKS_BASE_DIR = os.path.join(DATA_DIR, 'c_attacks')
UNTARGETED_ATTACKS_DIR = os.path.join(ATTACKS_BASE_DIR, 'untargeted', 'datasets')
UNTARGETED_EVAL_DIR = os.path.join(ATTACKS_BASE_DIR, 'untargeted', 'evaluations')

# Define report constants pointing inside ANALYSES_DIR
MY_DSPRITES_REPORT = os.path.join(ANALYSES_DIR, '01_my_dsprites_analysis.md')
MY_DSPRITES_CONTRAST_REPORT = os.path.join(ANALYSES_DIR, '02_my_dsprites_contrast_analysis.md')
MY_DSPRITES_SPLIT_REPORT = os.path.join(ANALYSES_DIR, '03_my_dsprites_split_analysis.md')
MY_DSPRITES_CONTRAST_SPLIT_REPORT = os.path.join(ANALYSES_DIR, '04_my_dsprites_contrast_split_analysis.md')
MODELS_TRAINING_REPORT = os.path.join(ANALYSES_DIR, '05_models_training_analysis.md')
UNTARGETED_ATTACKS_REPORT = os.path.join(ANALYSES_DIR, '06_untargeted_attacks_analysis.md')

# Define dataset constants pointing inside DATASETS_DIR
MY_DSPRITES = os.path.join(DATASETS_DIR, 'my_dsprites.npz')
MY_DSPRITES_SPLIT = os.path.join(DATASETS_DIR, 'my_dsprites_split.npz')
MY_DSPRITES_CONTRAST = os.path.join(DATASETS_DIR, 'my_dsprites_contrast.npz')
MY_DSPRITES_CONTRAST_SPLIT = os.path.join(DATASETS_DIR, 'my_dsprites_contrast_split.npz')


def get_model_paths(task, dataset, pool_type, loss_weight=None):
    """
    Constructs and returns a tuple of three file paths (model_path, history_path, dot_path)
    located inside MODELS_TRAINED_DIR.
    """
    base_name = f"{task}_{dataset}_{pool_type}"
    if loss_weight is not None:
        loss_weight_str = str(loss_weight).replace('.', 'p')
        base_name += f"_w{loss_weight_str}"
    model_path = os.path.join(MODELS_TRAINED_DIR, f"{base_name}_model.keras")
    history_path = os.path.join(MODELS_TRAINED_DIR, f"{base_name}_history.npy")
    dot_path = os.path.join(MODELS_TRAINED_DIR, f"{base_name}_dot.dot")
    return model_path, history_path, dot_path


def get_eval_paths(task, dataset, pool_type, loss_weight=None):
    """
    Constructs and returns a tuple of two file paths (primary_eval_path, secondary_eval_path)
    located inside MODELS_EVAL_DIR.
    """
    base_name = f"{task}_{dataset}_{pool_type}"
    if loss_weight is not None:
        loss_weight_str = str(loss_weight).replace('.', 'p')
        base_name += f"_w{loss_weight_str}"
    primary_eval_path = os.path.join(MODELS_EVAL_DIR, f"{base_name}_eval.json")
    secondary_eval_path = os.path.join(MODELS_EVAL_DIR, f"{base_name}_standard_test_eval.json")
    return primary_eval_path, secondary_eval_path


def get_latent_features_path(task, dataset, pool_type, loss_weight=None):
    """
    Constructs and returns the file path for latent features located inside MODELS_LATENT_DIR.
    """
    if task == 'single':
        base_name = f"single_{dataset}_{pool_type}_latent.npz"
    elif task == 'multi':
        if loss_weight is None:
            loss_weight = '1.0'
        loss_weight_str = str(loss_weight).replace('.', 'p')
        base_name = f"multi_{dataset}_{pool_type}_w{loss_weight_str}_latent.npz"
    else:
        raise ValueError(f"Invalid task: {task}. Must be 'single' or 'multi'")
    
    return os.path.join(MODELS_LATENT_DIR, base_name)


def get_untargeted_attack_path(dataset, pool_type, attack, norm, eps, loss_weight=None):
    """
    Constructs the file path for an untargeted adversarial attack dataset.
    """
    eps_str = str(eps).replace('.', 'p')
    if loss_weight is not None:
        loss_weight_str = str(loss_weight).replace('.', 'p')
        filename = f"untargeted_adv_{dataset}_{pool_type}_w{loss_weight_str}_{attack}_{norm}_eps{eps_str}.npz"
    else:
        filename = f"untargeted_adv_{dataset}_{pool_type}_{attack}_{norm}_eps{eps_str}.npz"
    return os.path.join(UNTARGETED_ATTACKS_DIR, filename)


def get_attack_eval_paths(dataset, pool_type, attack, norm, loss_weight=None):
    """
    Constructs the file path for untargeted attack evaluation results.
    Returns the CSV path string.
    """
    if loss_weight is not None:
        loss_weight_str = str(loss_weight).replace('.', 'p')
        base_filename = f"eval_{dataset}_{pool_type}_w{loss_weight_str}_{attack}_{norm}"
    else:
        base_filename = f"eval_{dataset}_{pool_type}_{attack}_{norm}"
    csv_path = os.path.join(UNTARGETED_EVAL_DIR, f"{base_filename}_details.csv.gz")
    return csv_path