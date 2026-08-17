"""
config.py — Central configuration for the IDS project.

Contains:
- NSL-KDD column names
- Attack type → category mappings
- File paths
"""

import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")

# =============================================================================
# NSL-KDD COLUMN NAMES
# =============================================================================
# 41 features + intrusion_type + difficulty_level
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'intrusion_type', 'difficulty'
]

# Categorical columns that need encoding
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Label column
LABEL_COL = 'intrusion_type'

# =============================================================================
# ATTACK CATEGORY MAPPING (NSL-KDD specific)
# =============================================================================
# Maps each specific attack name to one of 4 attack categories
ATTACK_CATEGORY_MAP = {
    # Normal
    'normal': 'Normal',

    # DoS — Denial of Service
    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'mailbomb': 'DoS',
    'apache2': 'DoS',
    'processtable': 'DoS',
    'udpstorm': 'DoS',

    # Probe — Surveillance/Scanning
    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'satan': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',

    # R2L — Remote to Local
    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',
    'snmpgetattack': 'R2L',
    'snmpguess': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'worm': 'R2L',

    # U2R — User to Root
    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R',
    'httptunnel': 'U2R',
    'ps': 'U2R',
    'sqlattack': 'U2R',
    'xterm': 'U2R',
}

# Category list (for consistent ordering)
CATEGORIES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# =============================================================================
# MODEL NAMES
# =============================================================================
MODEL_NAMES = [
    'Decision Tree',
    'Random Forest',
    'KNN',
    'Logistic Regression',
    'XGBoost',
    'Naive Bayes',
]

# =============================================================================
# RANDOM SEED (for reproducibility)
# =============================================================================
RANDOM_STATE = 42
