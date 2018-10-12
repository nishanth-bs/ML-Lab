import itertools
def sqr(x):
    return x**2

a = [1,2,34,5]

print(list(map(sqr,a)))
print(list(map(lambda x:''.join(i for i in x),list(itertools.permutations('abcd',4)))))
