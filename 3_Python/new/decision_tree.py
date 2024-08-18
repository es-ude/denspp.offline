import math

num_true = 6
num_false = 7


num_tot = num_true + num_false
if num_true is not 0:
    dev = num_true / num_tot
    C = -dev * math.log2(dev)
    print(C)
else:
    C = 0

if num_false is not 0:
    dev = num_false / num_tot
    D = -dev * math.log2(dev)
    print(D)
else:
    D = 0

print(C+D)
