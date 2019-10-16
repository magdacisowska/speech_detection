import numpy as np

file = open('inputs/giant.txt')
lines = file.readlines()
n_line = 0
for i in range(20, 100000):
    new_line = lines[i-20] + lines[i-19] + lines[i-18] + lines[i-17] + lines[i-16] + lines[i-15] + lines[i-14] +\
               lines[i-13] + lines[i-12] + lines[i-11] + lines[i-10] + lines[i-9] + lines[i-9] + lines[i-7] + \
               lines[i-6] + lines[i-5] + lines[i-4] + lines[i-3] + lines[i-2] + lines[i-1] + lines[i] + lines[i+1] +\
               lines[i+2] + lines[i+3] + lines[i+4] + lines[i+5] + lines[i+6] + lines[i+7] + lines[i+8] + lines[i+9] +\
               lines[i+10] + lines[i+11] + lines[i+12] + lines[i+13] + lines[i+14] + lines[i+15] + lines[i+16] +\
               lines[i+17] + lines[i+18] + lines[i+19] + lines[i+20]

    new_line = new_line.split()
    new_line = np.array(new_line).reshape((1, 533))
    new_line = new_line.astype(np.float)

    with open('inputs/giant_input.txt', 'ab') as f:
        np.savetxt(f, new_line)

    print(i)
