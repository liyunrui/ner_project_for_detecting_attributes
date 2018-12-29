#! /usr/bin/env python3
"""
Created on Oct 2 2018

Prepare training data for the following model.

@author: Ray

Mainly refer to the below:
    -https://github.com/guillaumegenthial/sequence_tagging
    -https://github.com/cristianoBY/Sequence-Tagging-NER-TensorFlow/blob/master/model/base_model.py
"""
import numpy as np
import os

# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

""".format(filename)
        super(MyIOError, self).__init__(message)

def makedirs(path_dir):
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)    

def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects
    Args:
        dataset: a iterator yielding words(str)
    Returns:
        a set of all the characters in the dataset 
    Notice:
        Here, we make the distinction between lowercase and uppercase, 
        for instance a and A are considered different
    """
    vocab_char = set()
    for words in dataset:
        for word in words:
            vocab_char.update(word)
    return vocab_char

def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors (xxx.txt)

    Returns:
        vocab_set: set() of strings
    """
    print("Building vocab...")
    vocab_set = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab_set.add(word)
    print("- done. {} tokens".format(len(vocab_set)))
    return vocab_set

def write_vocab(vocab, path_dir, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line (xxx.txt)

    """
    print("Writing vocab...")

    makedirs(path_dir)

    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab_and_return_word_to_id_dict(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index
    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                # give all the vocabualary in the intersection of our own corpurs and embedding vocabulary with a certain unique id
                word = word.strip() # remove strip
                d[word] = idx
    except IOError:
        raise MyIOError(filename)
    return d

def export_glove_vectors(vocab_dict, glove_filename, output_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab_dict: dictionary vocab_dict[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    
    Note:
        - Unknown words and Numbers have representations with values of zero. 
    """
    embeddings = np.zeros([len(vocab_dict), dim]) # initialize a zero
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0] 
            embedding = [float(x) for x in line[1:]]
            if word in vocab_dict:
                word_idx = vocab_dict[word]
                # np.array (by default) will make a copy of the object, while asarray will not unless necessary.
                embeddings[word_idx] = np.asarray(embedding) # it's better for memory.
            else:
                # 如果corpus的字沒有再交集裡面, 就不會給他打上id.
                pass
    # save several arrays into a single file in uncompressed .npz format.
    np.savez_compressed(output_filename, embeddings = embeddings)

def get_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]


    except IOError:
        raise MyIOError(filename)


# def get_chunk_type(tok, idx_to_tag):
#     """
#     Args:
#         tok: id of token, ex 4
#         idx_to_tag: dictionary {4: "B-PER", ...}

#     Returns:
#         tuple: "B", "PER"

#     """
#     tag_name = idx_to_tag[tok]
#     tag_class = tag_name.split('-')[0]
#     tag_type = tag_name.split('-')[-1]
#     return tag_class, tag_type


# def get_chunks(seq, tags):
#     """Given a sequence of tags, group entities and their position

#     Args:
#         seq: [4, 4, 0, 0, ...] sequence of labels
#         tags: dict. dict["O"] = 4

#     Returns:
#         list of (chunk_type, chunk_start, chunk_end)
#         chunk_start: it represents this tag start with index of this list.
#         chunk_end:it represents this tag end with index of this list + 1.
#     Example 1:
#         seq = [2,0,0]
#         tags = {'B-B': 2, 'I-B': 1, 'O': 0}
#         result = [('B', 0, 1)]
#     Example 2:
#         seq = [2,1,0]
#         tags = {'B-B': 2, 'I-B': 1, 'O': 0}
#         result = [('B', 0, 2)]
#     Example 3:
#         seq = [0,2,1]
#         tags = {'B-B': 2, 'I-B': 1, 'O': 0}
#         result = [('B', 1, 3)]
#     Example 4:
#         seq = [0,0,0]
#         tags = {'B-B': 2, 'I-B': 1, 'O': 0}
#         result = []
#     """
#     # setting
#     NONE = "O" # for get_chunks function

#     default = tags[NONE]
#     idx_to_tag = {idx: tag for tag, idx in tags.items()}
#     chunks = []
#     chunk_type, chunk_start = None, None
#     for i, tok in enumerate(seq):
#         # End of a chunk 1
#         if tok == default and chunk_type is not None:
#             # Add a chunk.
#             chunk = (chunk_type, chunk_start, i)
#             chunks.append(chunk)
#             chunk_type, chunk_start = None, None

#         # End of a chunk + start of a chunk!
#         elif tok != default:
#             tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
#             if chunk_type is None:
#                 chunk_type, chunk_start = tok_chunk_type, i
#             elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
#                 chunk = (chunk_type, chunk_start, i)
#                 chunks.append(chunk)
#                 chunk_type, chunk_start = tok_chunk_type, i
#         else:
#             pass

#     # end condition
#     if chunk_type is not None:
#         chunk = (chunk_type, chunk_start, len(seq))
#         chunks.append(chunk)

#     return chunks
