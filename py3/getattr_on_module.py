
import collections

# you can import module and get class from that module with getattr
dd = getattr(collections, 'defaultdict')(list)

dd['a'].append(1)
dd['a'].append(2)
print(dd)
