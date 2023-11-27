import re

with open("output/out2.txt") as f:
    lines = f.read().splitlines()
for line in lines:
    metrics = re.findall(r"epoch : (\d+) ,train loss : (\d+\.\d+) ,valid loss : (\d+\.\d+)", line)
    if len(metrics) > 0:
        epochs, train, val = metrics[0]
        print("\t".join(metrics[0]))