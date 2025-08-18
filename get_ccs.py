"""
Contrastive Concepts discovery based on pre-training dataset statistics.

This module finds contrastive concepts (negative examples) for given prompts using
CLIP embeddings and co-occurrence statistics.

The main workflow:
1. Encode input prompts using CLIP
2. Find nearest neighbors in a pre-computed vocabulary
3. Use co-occurrence statistics to find contrastive concepts
4. Optionally filter results with custom filters

    # Using command line
    python get_ccs.py --vocab_file laion400m_lemmatized.pkl --results_dir outputs --filter_file laion_concepts_map_v5.json --prompts_file prompts/cls_cityscapes.txt --gamma 0.01 --delta 0.8
"""

import argparse
import os.path
import pickle
import torch
from open_clip import get_tokenizer, create_model_from_pretrained
from imagenet_templates import imagenet_templates
from operator import itemgetter
import json
import numpy as np

# Words to remove from complement sets as they're too generic
REMOVE = ['image', 'photo', 'picture', 'view']


@torch.no_grad()
def encode_prompt(query_word, model, tokenizer, device):
    """
    Encode a single query word using CLIP text encoder with ImageNet templates.

    Args:
        query_word (str): The word to encode
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        device (str): Device to run computation on ('cuda' or 'cpu')

    Returns:
        torch.Tensor: Normalized CLIP text embedding (1D tensor)

    Note:
        Uses ImageNet templates (e.g., "a photo of a {}", "a picture of a {}")
        to create multiple text variations, then averages their embeddings.
    """
    # Create multiple text prompts using ImageNet templates
    query = tokenizer([temp.format(query_word) for temp in imagenet_templates]).to(device)

    # Encode all template variations
    feature = model.encode_text(query)

    # Normalize each template embedding
    feature /= feature.norm(dim=-1, keepdim=True)

    # Average across templates and normalize again
    feature = feature.mean(dim=0)
    feature /= feature.norm()

    return feature


def get_clip_similarities(texts, device='cuda'):
    """
    Compute CLIP embeddings for a list of texts and their pairwise similarities.

    Args:
        texts (list): List of text strings to encode
        device (str): Device to run computation on

    Returns:
        tuple: (similarity_matrix, embeddings)
            - similarity_matrix (torch.Tensor): Pairwise cosine similarities (N×N)
            - embeddings (torch.Tensor): CLIP embeddings for all texts (N×D)
    """
    # Load pre-trained CLIP model
    model, _ = create_model_from_pretrained("ViT-B-16", pretrained="laion2b_s34b_b88k")
    tokenizer = get_tokenizer("ViT-B-16")
    model = model.eval().to(device)

    # Encode all texts
    all_texts = torch.stack([encode_prompt(name, model, tokenizer, device) for name in texts])

    # Compute pairwise cosine similarities
    clip_scores = all_texts @ all_texts.T

    return clip_scores, all_texts


def get_cls_idx(path):
    """
    Load class names from a text file.

    Args:
        path (str): Path to file containing class names (one per line)

    Returns:
        list: List of class names with whitespace stripped

    Raises:
        AssertionError: If the file doesn't exist
    """
    assert os.path.isfile(path), f"File not found: {path}"

    with open(path, 'r') as f:
        name_sets = f.readlines()

    class_names = [name.strip() for name in name_sets]
    return class_names


def nearest_neighbour_match(prompt_embeds, vocab_embeds):
    """
    Find nearest neighbor matches between prompt embeddings and vocabulary embeddings.

    Args:
        prompt_embeds (torch.Tensor): Query embeddings (N×D)
        vocab_embeds (torch.Tensor): Vocabulary embeddings (M×D)

    Returns:
        torch.Tensor: Indices of nearest neighbors in vocabulary (N,)
    """
    # Compute similarity scores between prompts and vocabulary
    sim_scores = prompt_embeds @ vocab_embeds.T

    # Find the most similar vocabulary item for each prompt
    nn_indices = sim_scores.argmax(dim=1)

    return nn_indices


def get_negs(nn_indices, cooc_mat, gamma, delta, vocab_sim, keep_ids=None):
    """
    Find semantic complements based on co-occurrence statistics and similarity thresholds.

    Args:
        nn_indices (np.array): Indices of matched vocabulary items
        cooc_mat (np.array): Co-occurrence matrix (symmetric, M×M)
        gamma (float): Co-occurrence threshold (items with normalized co-occurrence > gamma are considered)
        delta (float): Similarity threshold (items with similarity < delta are kept as complements)
        vocab_sim (np.array): Pairwise similarity matrix for vocabulary items
        keep_ids (list, optional): Indices of vocabulary items to consider (filtering)

    Returns:
        list: List of boolean masks indicating complement items for each input

    Note:
        The algorithm finds items that:
        1. Co-occur frequently with the matched concept (normalized co-occurrence > gamma)
        2. Are semantically dissimilar (similarity < delta)
        This gives semantically related but contrasting concepts.
    """
    neg_indices = []

    # Create mask for items to consider (if filtering is applied)
    keep_mask = np.zeros(len(cooc_mat), dtype=bool)
    if keep_ids is not None:
        keep_mask[keep_ids] = True

    for i, mat_id in enumerate(nn_indices):
        # Normalize co-occurrences by diagonal (self co-occurrence)
        all_co = cooc_mat[mat_id] / cooc_mat[mat_id, mat_id]

        # Apply keep mask if provided
        all_co[~keep_mask] = 0

        # Find complements: high co-occurrence but low similarity
        neg_mask = (all_co > gamma) & (vocab_sim[mat_id] < delta)
        neg_indices.append(neg_mask)

    return neg_indices


def save_negs(neg_indices, vocab_names, file_path, file_path_bkg, prompts):
    """
    Save complement results to files in multiple formats.

    Args:
        neg_indices (list): List of boolean masks for complements
        vocab_names (list): Vocabulary concept names
        file_path (str): Main output file path
        file_path_bkg (str): Background output file path (with 'background' prefix)
        prompts (list): Original input prompts

    Creates three files:
        1. Main file: comma-separated complements per line
        2. Temp file: prompts with their complements
        3. Background file: 'background' + complements per line
    """
    avg = 0

    # Main output file
    with open(file_path, 'w') as f:
        for i, j_negs in enumerate(neg_indices):
            top_idx = j_negs.nonzero()[0]

            if len(top_idx) == 0:
                print(f'Warning, no complements for {prompts[i]}')
                neg_names = []
            elif len(top_idx) == 1:
                neg_names = [itemgetter(*top_idx.tolist())(vocab_names)]
            else:
                neg_names = list(itemgetter(*top_idx.tolist())(vocab_names))


            f.write(', '.join(neg_names))
            f.write('\n')
            print(f'{prompts[i]}: {neg_names} {len(neg_names)}')
            avg += len(neg_names)

    print(f'Average n complements: {avg / len(neg_indices)}')

    # Background output file (with 'background' prefix)
    with open(file_path_bkg, 'w') as f:
        for i, j_negs in enumerate(neg_indices):
            top_idx = j_negs.nonzero()[0]

            if len(top_idx) == 0:
                neg_names = []
            elif len(top_idx) == 1:
                neg_names = [itemgetter(*top_idx.tolist())(vocab_names)]
            else:
                neg_names = list(itemgetter(*top_idx.tolist())(vocab_names))

            neg_names = ['background'] + neg_names
            f.write(', '.join(neg_names))
            f.write('\n')


def filter_concepts(names, filter_dict):
    """
    Filter concept names based on a filter dictionary.

    Args:
        names (list): List of concept names
        filter_dict (dict): Dictionary where keys are concept names and values are booleans

    Returns:
        tuple: (filtered_indices, filtered_names)
            - filtered_indices (list): Indices of concepts that passed the filter
            - filtered_names (list): Names of concepts that passed the filter
    """
    filtered_ids = []
    f_names = []

    for i, name in enumerate(names):
        # Keep concept if it's in filter dict (with True value) and not in REMOVE list
        if filter_dict.get(name, False) and name not in REMOVE:
            filtered_ids.append(i)
            f_names.append(name)

    return filtered_ids, f_names


def read_file(file_path):
    """
    Read a file with comma-separated class names and create flat lists.

    Args:
        file_path (str): Path to input file

    Returns:
        tuple: (class_names, class_indices)
            - class_names (list): Flattened list of all class names
            - class_indices (list): Corresponding class indices for each name

    Note:
        Each line can contain multiple comma-separated names that belong to the same class.
    """
    with open(file_path, 'r') as f:
        name_sets = f.readlines()

    num_cls = len(name_sets)
    class_names, class_indices = list(), list()

    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]

    # Clean up newline characters
    class_names = [item.replace('\n', '') for item in class_names]

    return class_names, class_indices


def check_symmetric(matrix, rtol=1e-05, atol=1e-08):
    """
    Check if a matrix is symmetric within numerical tolerance.

    Args:
        matrix (np.array): Matrix to check
        rtol (float): Relative tolerance
        atol (float): Absolute tolerance

    Returns:
        bool: True if matrix is symmetric
    """
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)


def main(args):
    """
    Main function that orchestrates the complement discovery process.

    Args:
        args: Parsed command line arguments containing:
            - prompts_file: Path to file with input prompts
            - vocab_file: Path to vocabulary pickle file
            - gamma: Co-occurrence threshold
            - delta: Similarity threshold
            - filter_file: Optional filter file
            - results_dir: Directory to save results
            - device: Computation device
    """
    import time

    # Load input prompts
    prompts = get_cls_idx(args.prompts_file)

    print('Getting prompts clip embeddings')
    start = time.time()
    _, prompts_embeds = get_clip_similarities(prompts, device=args.device)
    print(f"CLIP embedding extraction: {time.time() - start:.2f}s")

    # Load vocabulary data
    assert os.path.isfile(args.vocab_file), f"Vocabulary file not found: {args.vocab_file}"
    print('Loading vocabulary file')
    with open(args.vocab_file, 'rb') as f:
        vocab_dict = pickle.load(f)

    vocab_embeds = vocab_dict['clip_embeds']
    vocab_names = vocab_dict['concept_names']
    coocmat = vocab_dict['cooc_mat']
    vocab_sims = vocab_dict['clip_sims']

    print('_' * 76)
    print('Matching the prompts with concepts.')
    start = time.time()
    nn_indices = nearest_neighbour_match(prompts_embeds, vocab_embeds.to(prompts_embeds.device))
    mapped_prompts = [vocab_names[i] for i in nn_indices.cpu().numpy()]
    print(f"Nearest neighbour matching: {time.time() - start:.2f}s")

    print('Matching completed')

    # Apply optional filtering
    filtered_ids = list(range(len(vocab_names)))
    if args.filter_file is not None:
        print('_' * 76)
        print('Running filtering')
        print(f'Number of concepts before filtering: {len(filtered_ids)}')

        with open(args.filter_file, 'r') as f:
            filter_dict = json.load(f)

        filtered_ids, _ = filter_concepts(vocab_names, filter_dict)
        print(f'Number of concepts after filtering: {len(filtered_ids)}')

    print('_' * 76)
    print('Finding semantic complements')
    neg_indices = get_negs(
        nn_indices.cpu().numpy(),
        coocmat,
        args.gamma,
        args.delta,
        vocab_sims,
        filtered_ids
    )

    # Generate output filename
    name_root = os.path.split(args.vocab_file)[-1].split('.')[0]
    name_pref = os.path.split(args.prompts_file)[-1].split('.')[0][4:]
    filename = f"{name_root}_{str(args.gamma).replace('.', '')}_{str(args.delta).replace('.', '')}_{name_pref}"

    if args.filter_file is not None:
        filename = 'filtered_v10_' + filename

    # Save mapping results
    mapping_file = os.path.join(args.results_dir, 'mapping_' + filename + '.txt')
    with open(mapping_file, 'w') as f:
        for org, mapped in zip(prompts, mapped_prompts):
            print(f'{org}: {mapped}')
            f.write(f'{org}: {mapped}\n')

    # Save complement results
    file_path = os.path.join(args.results_dir, filename + '.txt')
    file_path_bkg = os.path.join(args.results_dir, 'bkg_' + filename + '.txt')

    print('_' * 76)
    print(f'Saving complements to {file_path}')
    save_negs(neg_indices, vocab_names, file_path, file_path_bkg, prompts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find contrastive concepts for given prompts using CLIP embeddings and co-occurrence statistics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="Path to vocabulary pickle file containing co-occurence in the Laion dataset"
    )

    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Path to text file containing input prompts (one per line)"
    )

    parser.add_argument(
        "--filter_file",
        type=str,
        default=None,
        help="Path to JSON file for filtering vocabulary concepts"
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=0.8,
        help="Similarity threshold: concepts with similarity < delta are kept as contrastive concepts"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.01,
        help="Co-occurrence threshold: concepts with normalized co-occurrence > gamma are considered"
    )


    parser.add_argument(
        "--parts_filter_file",
        type=str,
        default=None,
        help="Additional filter file (currently unused)"
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.95,
        help="Additional threshold parameter for filtering (currently unused)"
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory to save output files'
    )

    parser.add_argument(
        '--device',
        default='cuda',
        help='Device for computation (cuda or cpu)'
    )


    args = parser.parse_args()
    main(args)