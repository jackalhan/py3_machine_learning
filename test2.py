test_cases = int(input())
numbers = [int(x) for x in input().strip().split()]

for number in numbers:
    # for value in range(1, number + 1):
    #     result = ''
    #     if value % 3 == 0 or value % 5 == 0:
    #         if value % 3 == 0:
    #             result += 'Fizz'
    #
    #         if value % 5 == 0:
    #             result += 'Buzz'
    #     else:
    #         result = value
    #
    #     print(result)

    print("\n".join(["Fizz"*(i%3==0)+"Buzz"*(i%5==0) or str(i) for i in range(1,number + 1)]))