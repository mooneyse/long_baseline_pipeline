#!/usr/bin/env python

print('this????????????????????????????????/')

def main(positional, optional=0):
        outdict = {}
        print 'File name: ', str(positional)

        # cast to types is a good idea/needed because parsets only work on strings
        # derivedval = int(optional) / 2
        #
        # # names in outdict get saved as 'optionalhalf.mapfile' and 'threspix.mapfile'
        # outdict['optionalhalf'] = derivedval
        # outdict['threshold'] = passthrough
        #
        # return outdict
        return 0

# def main (vis):
#     print(vis)
#     return 0
#
# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('vis', type=str, help='measurement set')
#
#     args = parser.parse_args()
#
#     main(vis=args.vis)
