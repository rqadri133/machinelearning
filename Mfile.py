nums = [8, 8, 8, 2, 1, 3]
k = 4

freq = {}

for n in nums:
    if n not in freq:
        freq[n] = 0
    freq[n] += 1

result = [x[0] for x in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]]

print(result)