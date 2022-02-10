import numpy as np
import torch

N_CLASSES = 150

def mask_to_subgrids(mask, cell_scale):
    """
    break WxH annotation array into a cell_scale x cell_scale vectors
    """
    num_elem_row, num_elem_col = int(mask.shape[0] / cell_scale), int(mask.shape[1] / cell_scale)
    res = []
    for h in range(cell_scale):
        for w in range(cell_scale):
            start_h = h * num_elem_row
            start_w = w * num_elem_col
            end_h = min((h+1)*num_elem_row, mask.shape[0])
            end_w = min((w+1)*num_elem_col, mask.shape[1])
            section = mask[start_h:end_h, start_w:end_w]
            res.append(section)
    return res

def unique_to_sparse(unique):
    """
    list of unique classes --> onehot sparse matrix
    """
    sparse = np.zeros((N_CLASSES))
    for num in unique:
        if num != 255:
            sparse[num] = 1
    return sparse

def arr_to_dist(onehot_mask):
    vec = onehot_mask.reshape(-1, N_CLASSES)
    dist = vec.sum(axis=0) / (vec.sum() + 1e-10)
    return dist

def vector_list_to_mat(vectors):
    """
    take list of vectors and stack them to a square matrix
    """
    n_rows = int(np.sqrt(len(vectors)))
    rows = []
    count = 0
    curr_row = []
    for i in range(len(vectors)):
        if count < n_rows:
            curr_row.append(vectors[i])
        if count == n_rows:
            count = 0
            rows.append(curr_row)
            curr_row = [vectors[i]]
        count += 1
    rows.append(curr_row)
    return np.asarray(rows)

def extract_mask_distributions(mask, head_sizes=[1], top_k=150):
    """
    mask: ground truth annotation (either BxWxH or WxH)
    head_sizes: list of scales at which to extract the distribution of pixels for each class
    top_k: limit # of classes, note even with k < C the distribution will add up to 1
    predicted_mask: if supplied, take the top classes from the predicted segmentation mask rather than ground truth annotation
    """
    if len(mask.size()) == 3: # if [B x W x H] rather than single sample [ W x H ]
        return [ extract_mask_distributions(mask[i], top_k=top_k, head_sizes=head_sizes) for i in range(mask.size()[0]) ]
    dist_labels = []
    for s in head_sizes:
        mat = extract_mask_distribution(mask, s)
        class_order = (-mat.flatten()).argsort()
        class_mask = np.where(np.in1d(np.arange(150), class_order[:top_k]), np.ones(150), np.zeros(150))
        class_mask = np.expand_dims(np.expand_dims(class_mask, -1), -1)
        masked_dist = class_mask * mat 
        masked_dist /= (np.sum(masked_dist, axis=None) + 1e-10)
    
        dist_labels.append(masked_dist)

    return dist_labels

def extract_mask_distribution(mask, scale=1):
    """
    Input: WxH integer-encoded label
    annotation --> pixel distribution at specified scales
    ignores background pixels (255)
    """ 
    onehot = (np.arange(255+1) == mask.numpy()[...,None]).astype(int)
    onehot_ignore = onehot[:,:,:N_CLASSES]
    if scale == 1: # special case
        mat = arr_to_dist(onehot_ignore)
        mat = np.expand_dims(mat, -1)
        mat = np.expand_dims(mat, -1)
    else:
        quadrants = mask_to_subgrids(onehot_ignore, scale)
        mat_vecs = [ arr_to_dist(m) for m in quadrants ]
        mat = vector_list_to_mat(mat_vecs).astype(np.float32)
        mat = mat.transpose(2, 0, 1)
    return mat


def extract_adjusted_distribution(gt_mask, predicted_mask, head_sizes=[1], top_k=150):
    """
    given ground truth annotation mask, and a trained segmentation network prediction,
    compute the distribution of the 'corrected' mask, s.t. pixels are equal to the 
    ground truth label if non-background, and predicted label if background

    this may offer a better training objective for the distribution of pixels for images
    with large portions of background class
    """
    gt_mask = gt_mask
    predicted_mask = predicted_mask
    corrected_mask = torch.where(gt_mask == 255, predicted_mask, gt_mask).cpu()
    corrected_distributions = [ extract_mask_distributions(corrected_mask[i], head_sizes=head_sizes) for i in range(corrected_mask.size()[0]) ]
    return corrected_distributions


def extract_mask_classes(mask, head_sizes=[1, 2, 3, 6]):
    """
    annotation mask --> set of head_sizes x head_sizes matrices with one-hot class labels
    encoding which classes are present in that region
    """
    classification_head_labels = []
    for s in head_sizes:
        if s == 1: # special case
            uniq = np.unique(mask)
            mat = unique_to_sparse(uniq)
            mat = np.expand_dims(mat, -1)
            mat = np.expand_dims(mat, -1)
        else:
            quadrants = mask_to_subgrids(mask, s)
            uniq_vectors = [ unique_to_sparse(np.unique(m)) for m in quadrants ]
            mat = vector_list_to_mat(uniq_vectors).astype(np.float32)
            mat = mat.transpose(2, 0, 1)
        classification_head_labels.append(mat)

    return classification_head_labels
