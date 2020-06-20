
def inverse(x):
    for i in range(1, 100000):
        if x * i % 11 == 1:
            return i
    
    return "uh oh!"