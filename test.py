input_queries = int(input())
for _ in range(input_queries):
    nums = input().split()
    x = int(nums[0])
    y = int(nums[1])
    pairs = []
    for i in range(x + 1):
        result = (x - i) ^ i
        if result == x:
            pairs.append((i, x-1))

    if not pairs:
        print(-1)
    else:
        pairs = sorted(pairs, key=lambda x:x[0])
        a , b = pairs[0]
        print(str(a), str(b))


