m,n = list(map(int,input("Enter m*n of first matrix\n").split()))
arr = [[0 for i in range(n)]for j in range(m)]
print(arr)
print("Enter n rows of the first matrix")
for i in range(m):
    row = list(map(int,input().split()))
    arr[i] = row

#print(arr)
n,p = n, int(input("Enter the number of columns"))
arrsecond = [[0 for i in range(p)]for j in range(n)]
print(arrsecond)
for i in range(n):
    row = list(map(int,input().strip().split()))
    arrsecond[i] = row

print(arrsecond)
res = [[0 for i in range(p)]for j in range(m)]
for i in range(m):
    for j in range(p):
        sum1 = 0
        for k in range(n):
            sum1 += arr[i][k] * arrsecond[k][j]
        res[i][j] = sum1
print(res)
