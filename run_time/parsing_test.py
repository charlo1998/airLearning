import os
os.sys.path.insert(0, os.path.abspath('../settings_folder'))

import json
from utils import *
import random
import settings
import msgs




def parse_data(file_name):
    with open(file_name, 'a+') as f:
        f.seek(0,0)
        lines = f.readlines()
        print(lines[-1])
        if lines[-1] != '}':
            print("need to add }!")
            f.write("\n}")
        else:
            print("already well formatted")


def main():
    file = os.path.join(settings.proj_root_path, "data", "DQN", "train_episodal_log.txt")
    parse_data(file)


if __name__ == "__main__":
    main()
