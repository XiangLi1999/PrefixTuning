import sys
import numpy as np
from collections import defaultdict
def read_file(path):
    result_dict = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if '.out' in line:
                curr_model = line
                result_dict[line] = [None, None, None]
            if 'rouge1=' in line:
                _, rouge1 = line.split('=')
                result_dict[curr_model][0] = float(rouge1)

            if 'rouge2=' in line:
                _, rouge2 = line.split('=')
                result_dict[curr_model][1] = float(rouge2)

            if 'rougeL=' in line:
                _, rougel = line.split('=')
                result_dict[curr_model][2] = float(rougel)
    # print(result_dict)
    return result_dict

def get_avg(result_dict):
    result = defaultdict(list)
    for file,val in result_dict.items():
        arguments = file.split('_')
        epoch_num = arguments[5]
        result[epoch_num].append(val)

    # print(result)
    for name, val in result.items():
        val = np.array(val)
        val_mean = val.mean(axis=0)
        print(name)
        # print(val)
        print(val_mean)




if __name__ == '__main__':
    path = sys.argv[1]
    result_dict = read_file(path)
    get_avg(result_dict)


