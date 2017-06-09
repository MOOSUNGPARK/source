def Nanugi(num):
    while num//2 == num/2:
        num /= 2
    while num//3 == num/3:
        num /= 3
    while num//5 == num/5 :
        num /= 5
    return num

# print(Nanugi(1500))


def Nanugi2(num,cnt):
    num_list = {1: 2, 2: 3, 3: 5}
    num_cnt = cnt
    nanum = num_list[num_cnt]

    if num//nanum == num/nanum:
        num /= nanum
        return Nanugi2(num, num_cnt)
    else :
        if num_cnt == 3:
            return num
        else :
            return Nanugi2(num, num_cnt+1)



from math import log
print(log(20))
print(log(2))

# print(Nanugi(1500))
# print(Nanugi2(1500,1))
# def Uglynum(cnt):
#     uglylist = []
#     num = 1
#     while len(uglylist) <= cnt:
#         if Nanugi(num) == 1 :
#             uglylist.append(num)
#         num+=1
#     return uglylist[cnt-1]
#
# if __name__ == '__main__':
#     while True:
#         cnt = int(input("숫자 입력(0 입력 시 exit) "))
#         if cnt == 0:
#             break
#         print(Uglynum(cnt))
