import numpy as np
import binary_adder

def x_gen():
    x = []
    temp_x = np.zeros(9)
    # A_0 -> A_3 B_0 -> B_3 C_0
    for A_0 in range(2):
        for A_1 in range(2):
            for A_2 in range(2):
                for A_3 in range(2):
                    for B_0 in range(2):
                        for B_1 in range(2):
                            for B_2 in range(2):
                                for B_3 in range(2):
                                    for C_0 in range(2):
                                        temp_x = [A_0, A_1, A_2, A_3, B_0, B_1, B_2, B_3, C_0]   
                                        x.append(temp_x)
    return x

def y_gen(x):
    y = []
    temp_y = np.zeros(5)
    # Y_0 -> Y_3 C_0
    for each in x:
        inputA = ''.join(str(i) for i in each[0:4])
        inputB = ''.join(str(i) for i in each[4:8])
        inputC = each[8]
        temp_y = binary_adder.binary_adder(inputA, inputB, inputC)
        temp_y = [int(a) for a in temp_y]
        y.append(temp_y)
    return y

def generate_data():
    x = x_gen()
    y = y_gen(x)
    x_shuffled = []
    y_shuffled = []
    shuffle_array = []
    for index, _ in enumerate(x):
        shuffle_array.append(x[index]+y[index])
    np.random.shuffle(shuffle_array)
    for arr in shuffle_array:
        x_shuffled.append(arr[0:9])
        y_shuffled.append(arr[9:14])
    return x_shuffled, y_shuffled

if __name__ == '__main__':
    x,y = generate_data()
    print(x[0], y[0])
