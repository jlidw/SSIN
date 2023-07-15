import torch
import random
import numpy as np
from tqdm import tqdm

# Multi seqs in one training batch;
# Return attn_mask to implement Shielded Attention instead of full-attention;
# Fixed: used the pkl data generated from preprocessing.py for training/testing, reduce the data redundancy.


# ------------------------------------- For training data ------------------------------------- #
def create_train_data(epoch, all_seq_data, invalid_masks, batch_size, masked_lm_prob, times=1, adj_attn_mask=None):
    """times: create how many times of the whole data"""

    data_num = len(all_seq_data)
    seq_len = all_seq_data.shape[1]
    all_indexes = list(range(data_num))

    max_pred_per_seq = int(seq_len * masked_lm_prob)

    batch_num = data_num // batch_size + 1

    with tqdm(total=batch_num*times, desc=f'Epoch {epoch}:') as pbar:
        for t in range(times):
            random.shuffle(all_indexes)  # for each masked round, shuffle the whole data

            for i in range(batch_num):
                end_idx = min(data_num, (i + 1) * batch_size)
                batch_indexes = all_indexes[i * batch_size: end_idx]

                masked_seq_list, masked_idx_list = [], []
                masked_labels_list, masked_label_weights_list, attn_mask_list = [], [], []
                for idx in batch_indexes:  # todo: modify batch, concatenate
                    seq_data = all_seq_data[idx]

                    if invalid_masks is not None:
                        invalid_mask = invalid_masks[idx]
                    else:
                        invalid_mask = None

                    masked_seq, masked_indexes, masked_labels, masked_label_weights, attn_mask = \
                        create_random_masked_data(seq_data, seq_len, invalid_mask, max_pred_per_seq, masked_lm_prob)

                    if adj_attn_mask is not None:
                        attn_mask = np.logical_and(attn_mask, adj_attn_mask).astype(int)

                    if masked_seq is not None:
                        masked_seq_list.append(masked_seq)
                        masked_idx_list.append(masked_indexes)
                        masked_labels_list.append(masked_labels)
                        masked_label_weights_list.append(masked_label_weights)
                        attn_mask_list.append(attn_mask)

                masked_seq_arr = np.array(masked_seq_list)
                masked_idx_arr = np.array(masked_idx_list)
                masked_labels_arr = np.array(masked_labels_list)
                masked_label_weights_arr = np.array(masked_label_weights_list)
                attn_mask_arr = np.array(attn_mask_list)

                # When all masked_seq with zero std, will generate empty batch
                # if masked_seq_arr.size != 0:
                yield [masked_seq_arr, masked_idx_arr, masked_labels_arr, masked_label_weights_arr, attn_mask_arr]
                pbar.update(1)


def create_random_masked_data(seq_data, seq_len, invalid_mask, max_pred_per_seq, masked_lm_prob=0.15):
    masked_seq = seq_data.copy()
    full_indexes = set(range(seq_len))

    if invalid_mask is not None:
        invalid_indexes = set(np.where(invalid_mask)[0])
    else:
        invalid_indexes = set()

    valid_indexes = full_indexes - invalid_indexes
    if len(valid_indexes) < 5:
        return None, None, None, None, None

    # masked_lm_prob = random.uniform(0.1, 0.3)
    n_pred = min(max_pred_per_seq, max(1, int(round(len(valid_indexes) * masked_lm_prob))))
    masked_indexes = random.sample((list(valid_indexes)), n_pred)
    masked_labels = list(seq_data[masked_indexes, :1])

    unmasked_indexes = list(valid_indexes - set(masked_indexes))
    mean_value = np.mean(seq_data[unmasked_indexes, :1])
    std_value = np.std(seq_data[unmasked_indexes, :1])

    # if std_value == 0:  # Discard the seqs: after normalization, will generate nan or inf value
    #     return None, None, None, None, None
    # masked_seq = (masked_seq - mean_value) / std_value  # normalization
    # masked_seq[masked_indexes, :1] = 0  # set masked nodes as 0

    # fixme: use this all zero sequence
    if std_value == 0:  # Discard the seqs: after normalization, will generate nan or inf value
        masked_seq = masked_seq - mean_value  # normalization
    else:
        masked_seq = (masked_seq - mean_value) / std_value  # normalization
    masked_seq[masked_indexes, :1] = 0  # set masked nodes as 0; this 0 denotes the mean value

    masked_label_weights = [1.0] * len(masked_labels)
    if max_pred_per_seq > n_pred:
        n_pad = max_pred_per_seq - n_pred
        masked_labels.extend([[0]] * n_pad)
        masked_indexes.extend([0] * n_pad)
        masked_label_weights.extend([0] * n_pad)

    attn_mask = get_attn_mask(seq_len, unmasked_indexes)

    masked_indexes = np.array(masked_indexes).astype(int)
    masked_label_weights = np.array(masked_label_weights).astype(float)
    attn_mask = np.array(attn_mask).astype(int)

    # fixme: std = 0
    # masked_labels = (np.array(masked_labels).astype(float) - mean_value) / std_value  # standardize label
    if std_value == 0:  # Discard the seqs: after normalization, will generate nan or inf value
        masked_labels = np.array(masked_labels).astype(float) - mean_value  # standardize label
    else:
        masked_labels = (np.array(masked_labels).astype(float) - mean_value) / std_value  # standardize label

    return masked_seq, masked_indexes, masked_labels, masked_label_weights, attn_mask


# ------------------------------------- For test data ------------------------------------- #
# one batch only includes one seq
def create_test_data(all_seq_data, invalid_masks, test_masks, all_timestamps, adj_attn_mask=None):
    data_num = len(all_seq_data)
    seq_len = all_seq_data.shape[1]
    all_indexes = range(data_num)

    with tqdm(total=data_num, desc=f'Testing:') as pbar:
        for idx in all_indexes:
            seq_data = all_seq_data[idx]
            timestamp = all_timestamps[idx]

            if invalid_masks is not None:
                invalid_mask = invalid_masks[idx]
            else:
                invalid_mask = None

            if test_masks.ndim == 2:
                test_mask = test_masks[idx]
            elif test_masks.ndim == 1:
                test_mask = test_masks

            # mean_value, std_value: real values
            masked_seq, masked_indexes, masked_labels, attn_mask, mean_value, std_value = \
                create_masked_data_by_idx(seq_data, seq_len, invalid_mask, test_mask)

            if adj_attn_mask is not None:
                attn_mask = np.logical_and(attn_mask, adj_attn_mask).astype(int)

            if masked_seq is not None:
                # convert to numpy, expand_dim: one batch includes one sequence
                masked_seq = np.expand_dims(np.array(masked_seq), axis=0).astype(float)
                masked_indexes = np.expand_dims(np.array(masked_indexes), axis=0).astype(int)
                attn_mask = np.expand_dims(np.array(attn_mask), axis=0).astype(float)
                timestamp_arr = np.array([timestamp] * len(masked_labels))

                # tensor: [batch_size=1, seq_len, in_dim], numpy: [seq_len]
                yield [masked_seq, masked_indexes, masked_labels,
                       attn_mask, mean_value, std_value, timestamp_arr]
                pbar.update(1)


def create_masked_data_by_idx(seq_data, seq_len, invalid_mask, test_mask):
    """For test data"""
    masked_seq = seq_data.copy()
    full_indexes = set(range(seq_len))

    if invalid_mask is not None:
        invalid_indexes = set(np.where(invalid_mask)[0])
    else:
        invalid_indexes = set()

    masked_indexes = np.where(test_mask)[0]  # test idx
    if masked_indexes.size == 0:  # Discard the seqs: if no valid test nodes; for aq dataset;
        return None, None, None, None, None, None

    masked_labels = seq_data[masked_indexes, :1]
    unmasked_indexes = list(full_indexes - invalid_indexes - set(masked_indexes))  # valid training nodes

    mean_value = np.mean(seq_data[unmasked_indexes, :1])
    std_value = np.std(seq_data[unmasked_indexes, :1])

    # if std_value == 0:  # Discard the seqs: after normalization, will generate nan or inf value
    #     return None, None, None, None, None, None
    # masked_seq = (masked_seq - mean_value) / std_value  # normalization
    # masked_seq[masked_indexes, :1] = 0  # set test nodes as 0

    # fixme: use this all zero sequence
    if std_value == 0:  # Discard the seqs: after normalization, will generate nan or inf value
        masked_seq = masked_seq - mean_value  # normalization
    else:
        masked_seq = (masked_seq - mean_value) / std_value  # normalization
    masked_seq[masked_indexes, :1] = 0  # set test nodes as 0

    # for test data: one batch only has one seq.
    attn_mask = get_attn_mask(seq_len, unmasked_indexes)

    return masked_seq, masked_indexes, masked_labels, attn_mask, mean_value, std_value


# ------------------------------------- util function ------------------------------------- #
def get_attn_mask(max_seq_len, unmasked_indexes):
    attn_vec = np.zeros(max_seq_len)

    # for each node, cut off the edge between it and invalid (masked or padded) nodes
    attn_vec[unmasked_indexes] = 1  # vector
    attn_mask = np.tile(attn_vec, (max_seq_len, 1))  # n*n matrix:

    # for each node (mainly for masked nodes), add itself edge for attention calculation
    eye_mask = np.eye(max_seq_len)

    attn_mask = np.logical_or(attn_mask, eye_mask).astype(int)

    return attn_mask


if __name__ == "__main__":
    pass

