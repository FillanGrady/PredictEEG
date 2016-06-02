import binascii
import sys
if __name__ == '__main__':
    """
    Annotations are formatted according to https://www.physionet.org/physiotools/wag/annot-5.htm
    In the MIT format
    In summary, they are two little-endian bytes
    Where the first 10 bits are the time after the last annotation, and the last 6 bits are the annotation type
    """
    with open(sys.argv[1]) as f:
        s = []
        while "6" not in s:
            s = f.read(1)
        while True:
            s = f.read(2)
            if len(s) != 2:
                break
            s = s[1] + s[0]
            b = format(int(binascii.hexlify(s), 16), '016b')
            t = b[0:10]
            a = b[10:]
            annotation = int(a, 2)
            time = int(t, 2)
            print "T: %s" % time
            print "A: %s" % annotation
            print "----------"