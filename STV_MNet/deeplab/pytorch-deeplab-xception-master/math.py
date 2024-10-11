import math
j = 0
k = 0
c = 0
for i in range(1, 72):
    a = 1 / math.cos(math.asin(i / 256))
    k = k + a
w = 2.5 / k
print(w)
for i in range(1, 219):
    e = 1 / math.cos(math.asin(i / 256))
    c = c + e
print(c)
l = c * w + 2.5
print(l)

for i in range(1, 72):
    t = 1 / math.cos(math.asin(i / 256))
    j = j + t
    if j * w > 1.2:
        print(i+256)
        break
g = math.cos(math.asin(35 / 256))
b = w * 6 * math.cos(math.asin(35 / 256))
print(b)
print(g)
