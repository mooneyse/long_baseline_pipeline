#!/usr/bin/env python


def main (vis):
    print(vis)
    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('vis', type=str, help='measurement set')

    args = parser.parse_args()

    main(vis=args.vis)
