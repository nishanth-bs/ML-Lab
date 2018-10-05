import string
s = list(input())
itr = 0
for i in s:
    if i.isalpha():
        s[itr] = i
        itr += 1
for i in range(itr-1, len(s)):
    s[i] = ''
print(''.join(s))
       
