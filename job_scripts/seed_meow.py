import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    args = parser.parse_args()
    if args.seed:
        print('i can haz seed')
    else:
        print('i can no haz seed')
