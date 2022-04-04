import os

def test_cpu():
    os.system('./run_cpu.sh')

def test_mlu():
    os.system('./run_mlu.sh')

if __name__ == '__main__':
    test_cpu()
    print('------------------------------------')
    test_mlu()