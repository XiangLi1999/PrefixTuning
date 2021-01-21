from rake_nltk import Rake
import  nltk
r = Rake()
# Uses stopwords for english from NLTK, and all puntuation characters by
# default
def get_keywords(inp):
    r.extract_keywords_from_text(inp)
    result = r.get_ranked_phrases()
    return result

swap = {'“':'\"', '”':'\"', '’':'\''}

if __name__ == '__main__':
    filename = '/u/scr/xlisali/contrast_LM/data_api/dataset/matching_debug_0.txt'
    outname = '/u/scr/xlisali/contrast_LM/data_api/dataset/matching_debug_0_kw.txt'
    line_lst = []
    key_lst = []
    out_handle = open(outname, 'w')
    with open(filename, 'r') as f:
        for line in f :
            line = line.strip().split('||')[1]
            # print(line)
            # print('here')
            line = line.split()
            for i, word in enumerate(line):
                if word in swap:
                    line[i] = swap[word]
                    print(line)

            line = ' '.join(line)

            result = get_keywords(line)
            line_lst.append(line)
            key_lst.append(result[:3])
            out_handle.write('{}||{}\n'.format(line, result[:3]))
    out_handle.close()

    # print(line_lst)
    # nltk.tokenize.sent_tokenize(line_lst[0])
    # r.extract_keywords_from_sentences(line_lst)
    # final = r.get_ranked_phrases()
    # # final = get_keywords(line_lst)
    # print(len(final), len(line_lst))
    # print(final[0], len(final), len(final[0]))



