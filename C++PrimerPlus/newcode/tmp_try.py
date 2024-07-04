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
