import numpy as np
import argparse
import skipthoughts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, default='Data/captions.txt',
                        help='caption file')
    parser.add_argument('--data_dir', type=str, default='Data',
                        help='Data Directory')

    args = parser.parse_args()
    with open(args.caption_file) as f:
        captions = f.read().split('\n')

    # captions : Text description of pictures stored in file sample_captions.txt
    captions = [cap for cap in captions if len(cap) > 0]
    print(captions)

    # create skipthoughts vectors
    model = skipthoughts.load_model()
    print('Creation of skipthought vectors : loading ....')
    caption_vectors = skipthoughts.encode(model, captions)
    print('Creation of skipthought vectors : DONE !')
    #print(caption_vectors)
    #print(np.shape(caption_vectors)).3

    # create tensor vectors with skipthought vectors as input
    print('Save skipthought vector : loading ....')
    np.save('skipvectors_2000.npy', caption_vectors)
    print('Save skipthought vector : DONE !')



if __name__ == '__main__':
    main()