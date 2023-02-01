def append_arr(arr, filepath):
    with open(filepath, "a") as f:
        for item in arr:
            f.write(str(item) + "\n")

def write_arr(arr, filepath):
    with open(filepath, "w") as f:
        for item in arr:
            f.write(str(item) + "\n")

def read_arr(filepath):
    res = []
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            res.append(float(line))

    return res
