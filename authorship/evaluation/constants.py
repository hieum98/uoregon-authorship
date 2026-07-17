"""HRS evaluation data path constants.
"""

# HRS_DIR = '/home/hieum/uonlp/hiatus_main/HRS'
HRS_DIR = '/gpfs/home/cuongp/hiatus-libs/HRS'
PERFORMER_CONFIG_PATH = 'TA2-performer-config.yaml'

HRS1_DIR = '/home/hieum/uonlp/avae/data/HRS/TA1'


HRS1_PATHS = {
    'HRS1.1': {
            'resample_queries': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.1/data/HRS1_english_long_sample-0_perGenre-HRS1.1_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.1/data/HRS1_english_long_sample-0_perGenre-HRS1.1_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.1/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.1_TA1_groundtruth.jsonl'
    },
    'HRS1.2': {
            'resample_queries': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.2/data/HRS1_english_long_sample-0_perGenre-HRS1.2_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.2/data/HRS1_english_long_sample-0_perGenre-HRS1.2_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.2/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.2_TA1_groundtruth.jsonl'
    },
    'HRS1.3': {
            'resample_queries': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.3/data/HRS1_english_long_sample-0_perGenre-HRS1.3_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.3/data/HRS1_english_long_sample-0_perGenre-HRS1.3_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.3/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.3_TA1_groundtruth.jsonl'
    },
    'HRS1.4': {
            'resample_queries': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.4/data/HRS1_english_long_sample-0_perGenre-HRS1.4_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.4/data/HRS1_english_long_sample-0_perGenre-HRS1.4_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.4/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.4_TA1_groundtruth.jsonl'
    },
    'HRS1.5': {
            'resample_queries': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.5/data/HRS1_english_long_sample-0_perGenre-HRS1.5_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.5/data/HRS1_english_long_sample-0_perGenre-HRS1.5_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS1_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.5/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.5_TA1_groundtruth.jsonl'
    },
    'en_cross_genre_long': {
            'resample_queries': f'{HRS1_DIR}/HRS1_english_long_sample-0_crossGenre/data/HRS1_english_long_sample-0_crossGenre_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS1_DIR}/HRS1_english_long_sample-0_crossGenre/data/HRS1_english_long_sample-0_crossGenre_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS1_DIR}/HRS1_english_long_sample-0_crossGenre/groundtruth/HRS1_english_long_sample-0_crossGenre_TA1_groundtruth.jsonl'
    },
}


HRS_PATHS = {
    'HRS2.1': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.1/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.1/data/HRS2_english_medium_sample-0_perGenre-HRS2.1_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.1/data/HRS2_english_medium_sample-0_perGenre-HRS2.1_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.1/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.1_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.1/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.1/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.1_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.1/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.1_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.1/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.1_TA2_candidate-labels.txt',
        }
    },
    'HRS2.2': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.2/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.2/data/HRS2_english_medium_sample-0_perGenre-HRS2.2_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.2/data/HRS2_english_medium_sample-0_perGenre-HRS2.2_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.2/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.2_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.2/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.2/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.2_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.2/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.2_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.2/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.2_TA2_candidate-labels.txt',
        }
    },
    'HRS2.3': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.3/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.3/data/HRS2_english_medium_sample-0_perGenre-HRS2.3_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.3/data/HRS2_english_medium_sample-0_perGenre-HRS2.3_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.3/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.3_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.3/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.3/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.3_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.3/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.3_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.3/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.3_TA2_candidate-labels.txt',
        }
    },
    'HRS2.4': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.4/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.4/data/HRS2_english_medium_sample-0_perGenre-HRS2.4_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.4/data/HRS2_english_medium_sample-0_perGenre-HRS2.4_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.4/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.4_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.4/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.4/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.4_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.4/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.4_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.4/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.4_TA2_candidate-labels.txt',
        }
    },
    'HRS2.5': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.5/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.5/data/HRS2_english_medium_sample-0_perGenre-HRS2.5_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.5/data/HRS2_english_medium_sample-0_perGenre-HRS2.5_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_perGenre-HRS2.5/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.5_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.5/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.5/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.5_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.5/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.5_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.5/groundtruth/HRS2_english_medium_sample-0_perGenre-HRS2.5_TA2_candidate-labels.txt',
        }
    },
    'en_cross_genre_medium': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_crossGenre/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_crossGenre/data/HRS2_english_medium_sample-0_crossGenre_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_crossGenre/data/HRS2_english_medium_sample-0_crossGenre_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA1/HRS2_english_medium_sample-0_crossGenre/groundtruth/HRS2_english_medium_sample-0_crossGenre_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_crossGenre/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_crossGenre/groundtruth/HRS2_english_medium_sample-0_crossGenre_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_crossGenre/groundtruth/HRS2_english_medium_sample-0_crossGenre_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_crossGenre/groundtruth/HRS2_english_medium_sample-0_crossGenre_TA2_candidate-labels.txt',
        }
    },
    'HRS1.1': {
        'TA1': {
            'data': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.1/data',
            'resample_queries': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.1/data/HRS1_english_long_sample-0_perGenre-HRS1.1_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.1/data/HRS1_english_long_sample-0_perGenre-HRS1.1_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.1/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.1_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.1/data',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.1/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.1_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.1/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.1_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.1/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.1_TA2_candidate-labels.txt',
        }
    },
    'HRS1.2': {
        'TA1': {
            'data': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.2/data',
            'resample_queries': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.2/data/HRS1_english_long_sample-0_perGenre-HRS1.2_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.2/data/HRS1_english_long_sample-0_perGenre-HRS1.2_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.2/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.2_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.2/data',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.2/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.2_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.2/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.2_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.2/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.2_TA2_candidate-labels.txt',
        }
    },
    'HRS1.3': {
        'TA1': {
            'data': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.3/data',
            'resample_queries': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.3/data/HRS1_english_long_sample-0_perGenre-HRS1.3_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.3/data/HRS1_english_long_sample-0_perGenre-HRS1.3_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.3/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.3_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.3/data',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.3/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.3_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.3/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.3_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.3/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.3_TA2_candidate-labels.txt',
        }
    },
    'HRS1.4': {
        'TA1': {
            'data': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.4/data',
            'resample_queries': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.4/data/HRS1_english_long_sample-0_perGenre-HRS1.4_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.4/data/HRS1_english_long_sample-0_perGenre-HRS1.4_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.4/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.4_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.4/data',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.4/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.4_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.4/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.4_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.4/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.4_TA2_candidate-labels.txt',
        }
    },
    'HRS1.5': {
        'TA1': {
            'data': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.5/data',
            'resample_queries': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.5/data/HRS1_english_long_sample-0_perGenre-HRS1.5_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.5/data/HRS1_english_long_sample-0_perGenre-HRS1.5_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_perGenre-HRS1.5/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.5_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.5/data',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.5/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.5_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.5/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.5_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.5/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.5_TA2_candidate-labels.txt',
        }
    },
    'en_cross_genre_long': {
        'TA1': {
            'data': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_crossGenre/data',
            'resample_queries': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_crossGenre/data/HRS1_english_long_sample-0_crossGenre_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_crossGenre/data/HRS1_english_long_sample-0_crossGenre_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long/TA1/HRS1_english_long_sample-0_crossGenre/groundtruth/HRS1_english_long_sample-0_crossGenre_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_crossGenre/data',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_crossGenre/groundtruth/HRS1_english_long_sample-0_crossGenre_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_crossGenre/groundtruth/HRS1_english_long_sample-0_crossGenre_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/HRS1_english_long/TA2/HRS1_english_long_sample-0_crossGenre/groundtruth/HRS1_english_long_sample-0_crossGenre_TA2_candidate-labels.txt',
        }
    },
    'HRS2.101': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_perGenre-HRS2.101/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_perGenre-HRS2.101/data/HRS2_russian_long_sample-0_perGenre-HRS2.101_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_perGenre-HRS2.101/data/HRS2_russian_long_sample-0_perGenre-HRS2.101_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_perGenre-HRS2.101/groundtruth/HRS2_russian_long_sample-0_perGenre-HRS2.101_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_perGenre-HRS2.101/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_perGenre-HRS2.101/groundtruth/HRS2_russian_long_sample-0_perGenre-HRS2.101_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_perGenre-HRS2.101/groundtruth/HRS2_russian_long_sample-0_perGenre-HRS2.101_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_perGenre-HRS2.101/groundtruth/HRS2_russian_long_sample-0_perGenre-HRS2.101_TA2_candidate-labels.txt',
        }
    },
    'HRS2.102': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_perGenre-HRS2.102/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_perGenre-HRS2.102/data/HRS2_russian_long_sample-0_perGenre-HRS2.102_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_perGenre-HRS2.102/data/HRS2_russian_long_sample-0_perGenre-HRS2.102_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_perGenre-HRS2.102/groundtruth/HRS2_russian_long_sample-0_perGenre-HRS2.102_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_perGenre-HRS2.102/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_perGenre-HRS2.102/groundtruth/HRS2_russian_long_sample-0_perGenre-HRS2.102_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_perGenre-HRS2.102/groundtruth/HRS2_russian_long_sample-0_perGenre-HRS2.102_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_perGenre-HRS2.102/groundtruth/HRS2_russian_long_sample-0_perGenre-HRS2.102_TA2_candidate-labels.txt',
        }
    },
    'HRS2.103': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_perGenre-HRS2.103/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_perGenre-HRS2.103/data/HRS2_russian_long_sample-0_perGenre-HRS2.103_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_perGenre-HRS2.103/data/HRS2_russian_long_sample-0_perGenre-HRS2.103_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_perGenre-HRS2.103/groundtruth/HRS2_russian_long_sample-0_perGenre-HRS2.103_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_perGenre-HRS2.103/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_perGenre-HRS2.103/groundtruth/HRS2_russian_long_sample-0_perGenre-HRS2.103_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_perGenre-HRS2.103/groundtruth/HRS2_russian_long_sample-0_perGenre-HRS2.103_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_perGenre-HRS2.103/groundtruth/HRS2_russian_long_sample-0_perGenre-HRS2.103_TA2_candidate-labels.txt',
        }
    },
    'HRS2.104': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA1/HRS2_russian_medium_sample-0_perGenre-HRS2.104/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA1/HRS2_russian_medium_sample-0_perGenre-HRS2.104/data/HRS2_russian_medium_sample-0_perGenre-HRS2.104_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA1/HRS2_russian_medium_sample-0_perGenre-HRS2.104/data/HRS2_russian_medium_sample-0_perGenre-HRS2.104_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA1/HRS2_russian_medium_sample-0_perGenre-HRS2.104/groundtruth/HRS2_russian_medium_sample-0_perGenre-HRS2.104_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA2/HRS2_russian_medium_sample-0_perGenre-HRS2.104/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA2/HRS2_russian_medium_sample-0_perGenre-HRS2.104/groundtruth/HRS2_russian_medium_sample-0_perGenre-HRS2.104_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA2/HRS2_russian_medium_sample-0_perGenre-HRS2.104/groundtruth/HRS2_russian_medium_sample-0_perGenre-HRS2.104_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA2/HRS2_russian_medium_sample-0_perGenre-HRS2.104/groundtruth/HRS2_russian_medium_sample-0_perGenre-HRS2.104_TA2_candidate-labels.txt',
        }
    },
    'HRS2.105': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA1/HRS2_russian_medium_sample-0_perGenre-HRS2.105/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA1/HRS2_russian_medium_sample-0_perGenre-HRS2.105/data/HRS2_russian_medium_sample-0_perGenre-HRS2.105_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA1/HRS2_russian_medium_sample-0_perGenre-HRS2.105/data/HRS2_russian_medium_sample-0_perGenre-HRS2.105_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA1/HRS2_russian_medium_sample-0_perGenre-HRS2.105/groundtruth/HRS2_russian_medium_sample-0_perGenre-HRS2.105_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA2/HRS2_russian_medium_sample-0_perGenre-HRS2.105/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA2/HRS2_russian_medium_sample-0_perGenre-HRS2.105/groundtruth/HRS2_russian_medium_sample-0_perGenre-HRS2.105_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA2/HRS2_russian_medium_sample-0_perGenre-HRS2.105/groundtruth/HRS2_russian_medium_sample-0_perGenre-HRS2.105_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA2/HRS2_russian_medium_sample-0_perGenre-HRS2.105/groundtruth/HRS2_russian_medium_sample-0_perGenre-HRS2.105_TA2_candidate-labels.txt',
        }
    },
    'ru_cross_genre_long': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_crossGenre/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_crossGenre/data/HRS2_russian_long_sample-0_crossGenre_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_crossGenre/data/HRS2_russian_long_sample-0_crossGenre_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA1/HRS2_russian_long_sample-0_crossGenre/groundtruth/HRS2_russian_long_sample-0_crossGenre_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_crossGenre/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_crossGenre/groundtruth/HRS2_russian_long_sample-0_crossGenre_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_crossGenre/groundtruth/HRS2_russian_long_sample-0_crossGenre_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_long/TA2/HRS2_russian_long_sample-0_crossGenre/groundtruth/HRS2_russian_long_sample-0_crossGenre_TA2_candidate-labels.txt',
        }
    },
    'ru_cross_genre_medium': {
        'TA1': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA1/HRS2_russian_medium_sample-0_crossGenre/data',
            'resample_queries': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA1/HRS2_russian_medium_sample-0_crossGenre/data/HRS2_russian_medium_sample-0_crossGenre_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA1/HRS2_russian_medium_sample-0_crossGenre/data/HRS2_russian_medium_sample-0_crossGenre_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA1/HRS2_russian_medium_sample-0_crossGenre/groundtruth/HRS2_russian_medium_sample-0_crossGenre_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA2/HRS2_russian_medium_sample-0_crossGenre/data',
            'ground_truth': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA2/HRS2_russian_medium_sample-0_crossGenre/groundtruth/HRS2_russian_medium_sample-0_crossGenre_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA2/HRS2_russian_medium_sample-0_crossGenre/groundtruth/HRS2_russian_medium_sample-0_crossGenre_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/11_22_24/HRS_evaluation_samples/HRS2_russian_medium/TA2/HRS2_russian_medium_sample-0_crossGenre/groundtruth/HRS2_russian_medium_sample-0_crossGenre_TA2_candidate-labels.txt',
        }
    },
}


HRS3_PATHS = {
    'HRS3.1':{
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1/data/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1/data/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.1_TA2_candidate-labels.txt',
        }
    },
    'HRS3.2': {
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2/data/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2/data/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.2_TA2_candidate-labels.txt',
        }
    },
    'HRS3.3': {
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3/data/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3/data/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.3_TA2_candidate-labels.txt',
        }
    },
    'HRS3.4': {
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4/data/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4/data/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.4_TA2_candidate-labels.txt',
        }
    },
    'HRS3.5': {
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5/data/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5/data/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5/groundtruth/HRS3_10-24-25_english_short_sample-0_perGenre-HRS3.5_TA2_candidate-labels.txt',
        } 
    },
    'en_cross_genre_short': {
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_crossGenre/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_crossGenre/data/HRS3_10-24-25_english_short_sample-0_crossGenre_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_crossGenre/data/HRS3_10-24-25_english_short_sample-0_crossGenre_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA1/HRS3_10-24-25_english_short_sample-0_crossGenre/groundtruth/HRS3_10-24-25_english_short_sample-0_crossGenre_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_crossGenre/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_crossGenre/groundtruth/HRS3_10-24-25_english_short_sample-0_crossGenre_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_crossGenre/groundtruth/HRS3_10-24-25_english_short_sample-0_crossGenre_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_english_short/TA2/HRS3_10-24-25_english_short_sample-0_crossGenre/groundtruth/HRS3_10-24-25_english_short_sample-0_crossGenre_TA2_candidate-labels.txt',
        }
    },
    'HRS3.301': {
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_long/TA1/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_long/TA1/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301/data/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_long/TA1/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301/data/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_long/TA1/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301/groundtruth/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_long/TA2/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_long/TA2/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301/groundtruth/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_long/TA2/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301/groundtruth/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_long/TA2/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301/groundtruth/HRS3_10-24-25_chinese_long_sample-0_perGenre-HRS3.301_TA2_candidate-labels.txt',
        }
    },
    'HRS3.302': {
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302/data/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302/data/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302/groundtruth/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA2/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA2/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302/groundtruth/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA2/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302/groundtruth/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA2/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302/groundtruth/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.302_TA2_candidate-labels.txt',
        }
    },
    'HRS3.303': {
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA1/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA1/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303/data/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA1/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303/data/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA1/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303/groundtruth/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA2/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA2/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303/groundtruth/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA2/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303/groundtruth/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA2/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303/groundtruth/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.303_TA2_candidate-labels.txt',
        }
    },
    'HRS3.304': {
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304/data/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304/data/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304/groundtruth/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA2/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA2/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304/groundtruth/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA2/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304/groundtruth/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA2/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304/groundtruth/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.304_TA2_candidate-labels.txt',
        }
    },
    'HRS3.305': {
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_perGenre-HRS3.305/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA1/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305/data/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA1/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305/data/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA1/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305/groundtruth/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA2/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA2/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305/groundtruth/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA2/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305/groundtruth/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA2/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305/groundtruth/HRS3_10-24-25_chinese_short_sample-0_perGenre-HRS3.305_TA2_candidate-labels.txt',
        }
    },
    'zh_cross_genre_medium': {
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_crossGenre/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_crossGenre/data/HRS3_10-24-25_chinese_medium_sample-0_crossGenre_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_crossGenre/data/HRS3_10-24-25_chinese_medium_sample-0_crossGenre_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA1/HRS3_10-24-25_chinese_medium_sample-0_crossGenre/groundtruth/HRS3_10-24-25_chinese_medium_sample-0_crossGenre_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA2/HRS3_10-24-25_chinese_medium_sample-0_crossGenre/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA2/HRS3_10-24-25_chinese_medium_sample-0_crossGenre/groundtruth/HRS3_10-24-25_chinese_medium_sample-0_crossGenre_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA2/HRS3_10-24-25_chinese_medium_sample-0_crossGenre/groundtruth/HRS3_10-24-25_chinese_medium_sample-0_crossGenre_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_medium/TA2/HRS3_10-24-25_chinese_medium_sample-0_crossGenre/groundtruth/HRS3_10-24-25_chinese_medium_sample-0_crossGenre_TA2_candidate-labels.txt',
        }
    },
    'zh_cross_genre_short': {
        'TA1': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA1/HRS3_10-24-25_chinese_short_sample-0_crossGenre/data',
            'resample_queries': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA1/HRS3_10-24-25_chinese_short_sample-0_crossGenre/data/HRS3_10-24-25_chinese_short_sample-0_crossGenre_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA1/HRS3_10-24-25_chinese_short_sample-0_crossGenre/data/HRS3_10-24-25_chinese_short_sample-0_crossGenre_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA1/HRS3_10-24-25_chinese_short_sample-0_crossGenre/groundtruth/HRS3_10-24-25_chinese_short_sample-0_crossGenre_TA1_groundtruth.jsonl'
        },
        'TA2': {
            'data': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA2/HRS3_10-24-25_chinese_short_sample-0_crossGenre/data',
            'ground_truth': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA2/HRS3_10-24-25_chinese_short_sample-0_crossGenre/groundtruth/HRS3_10-24-25_chinese_short_sample-0_crossGenre_TA2_groundtruth.npy',
            'ground_truth_query_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA2/HRS3_10-24-25_chinese_short_sample-0_crossGenre/groundtruth/HRS3_10-24-25_chinese_short_sample-0_crossGenre_TA2_query-labels.txt',
            'ground_truth_candidate_labels': f'{HRS_DIR}/10_27_25/HRS_eval/HRS3_chinese_short/TA2/HRS3_10-24-25_chinese_short_sample-0_crossGenre/groundtruth/HRS3_10-24-25_chinese_short_sample-0_crossGenre_TA2_candidate-labels.txt',
        }
    },
}

ALL_HRS_PATHS = {**HRS_PATHS, **HRS3_PATHS}

GENRE_GROUPS = {
    "en_short": ["HRS3.1", "HRS3.2", "HRS3.3", "HRS3.4", "HRS3.5", "en_cross_genre_short"],
    "zh": [
        "HRS3.301", "HRS3.302", "HRS3.303", "HRS3.304", "HRS3.305",
        "zh_cross_genre_medium", "zh_cross_genre_short",
    ],
    "en_medium": ["HRS2.1", "HRS2.2", "HRS2.3", "HRS2.4", "HRS2.5", "en_cross_genre_medium"],
    "en_long": ["HRS1.1", "HRS1.2", "HRS1.3", "HRS1.4", "HRS1.5", "en_cross_genre_long"],
    "ru": [
        "HRS2.101", "HRS2.102", "HRS2.103", "HRS2.104", "HRS2.105",
        "ru_cross_genre_long", "ru_cross_genre_medium",
    ],
}