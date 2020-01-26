import pandas as pd


filename = '/home/akira/Project/pl-svo/bin/relPose-targetodom.csv'
fileout = filename + 'noscale'

arr = pd.read_csv(filename, header=None).values
with open(fileout, 'w') as f:
    for line in arr:
        time = line[0]
        dx = line[1]
        dy = line[2]
        dtheta = line[3]
        scale = (dx ** 2 + dy ** 2) ** (1 / 2)
        if scale>0:
            dx = dx / scale
            dy = dy / scale
        s = '{},{:.6E},{:.6E},{:.6E}\n'.format(time, dx, dy, dtheta)
        f.write(s)