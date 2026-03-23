records = [
    {"student":"A","score":80,"weight":0.2},
    {"student":"A","score":90,"weight":0.8},
    {"student":"B","score":70,"weight":0.5},
    {"student":"B","score":100,"weight":0.5},
    {"student":"C","score":60,"weight":1.0}
]

weighted_sum = {}
total_weight = {}

for r in records:

    student = r["student"]
    score = r["score"]
    weight = r["weight"]

    if student not in weighted_sum:
        weighted_sum[student] = 0
        total_weight[student] = 0

    weighted_sum[student] += score * weight
    total_weight[student] += weight


for student in weighted_sum:

    mean = weighted_sum[student] / total_weight[student]

    print(student,"->",mean)

    