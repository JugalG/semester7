def trail(n):
    bits = 0
    x = n

    if x:
        while (x & 1) == 0:
            bits += 1
            x >>= 1
    return bits

def decToBinary(n):
    binaryNum = [0] * 32
    i = 0
    while n > 0:
        binaryNum[i] = n % 2
        n = n // 2
        i += 1

    ans = ""
    for j in range(i - 1, -1, -1):
        ans += str(binaryNum[j])
    return ans

def main():
    n = int(input())
    a = []
    bin = []
    mx = 0

    for _ in range(n):
        a_i = int(input())
        qs = (6 * a_i + 1) % 5
        ans = decToBinary(a_i)
        bin.append(ans)

    for i in range(len(bin)):
        j = int(bin[i], 2)
        maxi = trail(j)
        mx = max(mx, maxi)

    print(2**mx)


main()