from packaging import version
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--c', type=str, help='current version')
parser.add_argument('--l', type=str, help='latest deployed version')
args = parser.parse_args()
if not args.l:
    print(True)
else:
    print(version.parse(args.c) >= version.parse(args.l))
