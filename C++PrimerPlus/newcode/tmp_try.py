"""
A = "abczhdshoozjpzjoppzhiahso"
B = "shsojosmxhposhoozjapsjnxjhp"

tmp = ""
tmp_str = ""

len_a = len(A)
len_b = len(B)
print("len_a = ", len_a, "len_b = ", len_b)

lenth = len_a if len_a < len_b else len_b

print(lenth)

for i in range(lenth):
    for j in range(i+1, lenth - i):
        if len_a > len_b:
            short_str = B
            long_str = A 
        else: 
            short_str = A
            long_str = B
        tmp = short_str[i:j]
        if tmp in long_str and len(tmp) > len(tmp_str):
            tmp_str = tmp
print(tmp_str)


a = " the fist input tmpahaajojoj"
tmp = a.split(" ")[-1]
print(len(tmp))


a = "tmpzh angsu ibiAHSSOH  OJ123679001524646182979  "
b = "6"
print(a.lower())
num = 0
print(a.lower().count(' '))
for i in a.lower():
    if b.lower() == i:
        num = num + 1
print(num) 


import random
num = 20
num_list = [2, 7, 6, 3, 4, 5, 7, 2, 2, 3, 8, 10, 5, 4, 2]
# for i in range(20):
#     num_list.append(random.randint(0,500))
print(set(num_list))


a = "tmpzh angsu ibiAHSSOH  OJ123679001524646182979  zj"
print(len(a))
l_t = len(a) // 8
t_l = l_t+1 if (len(a) % 8) else  l_t
for i in range(t_l):
    if i == t_l-1:
        print(a[i*8:(i+1)*8] + (8-len(a[i*8:(i+1)*8]))*'0')
    else:
        print(a[i*8:(i+1)*8])
"""

print(oct(10))
print(hex(10))
print(bin(10))
print(0x12bacf)
print(0x1a)
print(0xa1)
tmp_str = '0x12bacf'
dict_hex = {'a': 10, 'b': 11, 'c': 12, 'd':13, 'e': 14, 'f': 15}
print(tmp_str[2:])
tmp_len = len(tmp_str[2:])-1
res = 0
for i in tmp_str[2:]:
    if i in dict_hex:
        n = dict_hex[i]
    else:
        n = i
    res += int(n) * 16**(tmp_len)
    tmp_len = tmp_len-1
print(res)
