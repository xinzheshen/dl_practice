import numpy as np
import w2v_utils



if __name__ == '__main__':

    words, word_to_vec_map = w2v_utils.read_glove_vecs('data/glove.6B.50d.txt')
    print(word_to_vec_map['hello'])