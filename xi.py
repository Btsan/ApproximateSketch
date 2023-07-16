import torch
# CW2B and EH3

def hash31(a, b, x):
    """ Adapted from MassDal: http://www.cs.rutgers.edu/~muthu/massdal-code-index.html
    Computes Carter-Wegman (CW) hash with Mersenne trick*/
    """
    res = a * x + b
    return ((res >> 31) + res) & 2147483647

def seq_xor(x):
    """ Computes parity bit of the bits of an integer*/
    """
    x ^= (x >> 16)
    x ^= (x >> 8)
    x ^= (x >> 4)
    x ^= (x >> 2)
    x ^= (x >> 1)
    return (x & 1)

def EH3(i0, I1, j):
    """ +-1 random variables, 3-wise independent schemes
    """
    mask = 0xAAAAAAAA
    p_res = (I1 & j) ^ (j & (j<<1) & mask)
    return torch.where(((i0 ^ seq_xor(p_res)) & (1 == 1)) != 0, 1, -1)
    # return torch.where(((i0 ^ seq_xor(p_res)) & 1 == 1), 1, -1)

def CW2B(a, b, x, M):
    """b-valued random variables 2-wise CW scheme
    """
    p_res = hash31(a, b, x)
    res = p_res % M;
    return res;

class B_Xi(object):
    def __init__(self, B, I1=2**32-1, I2=2**32-1, preprocess=True):
        super(B_Xi, self).__init__()
        """ hash to B buckets
        """
        self.num_buckets = B
        seeds = torch.tensor([I1, I2], dtype=torch.int64)
        if preprocess:
            k_mask = 0xffffffff
            args = torch.tensor([I1, I2], dtype=torch.int64)
            seeds[0] = ((args[0] << 16)^(args[1] & 0x0000ffff)) & k_mask
            args[0] = (36969*(args[0] & 0x0000ffff)) + ((args[0])>>16)
            args[1] = (18000*(args[1] & 0x0000ffff)) + ((args[1])>>16)
            seeds[1] = ((args[0] << 16)^(args[1] & 0x0000ffff)) & k_mask
        self.seeds = seeds

    def element(self, j):
        return CW2B(*(self.seeds), j, self.num_buckets)
    def __call__(self, j):
        return self.element(j)
    def __str__(self):
        return "{}-wise xi({}, {})".format(self.num_buckets, *(self.seeds))
    def __repr__(self):
        return str(self)

class Xi(object):
    def __init__(self, I1=2**32-1, I2=2**32-1, preprocess=True):
        super(Xi, self).__init__()
        """ hash to pos or neg 1
        """
        seeds = torch.tensor([I1, I2], dtype=torch.int64)
        if preprocess:
            k_mask = 0xffffffff
            args = torch.tensor([I1, I2], dtype=torch.int64)
            seeds[0] = ((args[0] << 16)^(args[1] & 0x0000ffff)) & k_mask
            args[0] = (36969*(args[0] & 0x0000ffff)) + ((args[0])>>16)
            args[1] = (18000*(args[1] & 0x0000ffff)) + ((args[1])>>16)
            seeds[1] = ((args[0] << 16)^(args[1] & 0x0000ffff)) & k_mask
        self.seeds = seeds

    def element(self, j):
        return EH3(*(self.seeds), j)
    def __call__(self, j):
        return self.element(j)
    def __str__(self):
        return "+-1 xi({}, {})".format(*(self.seeds))
    def __repr__(self):
        return str(self)

if __name__ == '__main__':
    import random
    random.seed(2 ** 31 - 1)
    interval = 100000

    print("seeding test...")
    for _ in range(10000):
        random.randint(0, 2**31-1)

    xi = Xi(random.randint(1, 2**32-1), random.randint(1, 2**32-1))
    count = 0
    for i in range(interval):
        res = xi(i)
        assert res in (-1, 1), (res, xi)
        count += res
    print("{} sum over range({}): {}".format(xi, interval, count.item()))

    b_xi = B_Xi(100, random.randint(1, 2**32-1), random.randint(1, 2**32-1))
    total = 0
    for i in range(interval):
        res = b_xi(i)
        assert res >= 0, (res, i)
        total += res + 1
    print("{} average {} / {} = {}".format(b_xi, total, interval, total/interval))


    b_xi_1 = B_Xi(10, random.randint(1, 2**32-1), random.randint(1, 2**32-1))
    b_xi_2 = B_Xi(10, random.randint(1, 2**32-1), random.randint(1, 2**32-1))
    hits = 0
    for i in range(interval):
        hits += (b_xi_1(i) == b_xi_2(i))
    print("({}, {}) hit {} / {} = {} rate".format(b_xi_1, b_xi_2, hits, interval, hits/interval))

    print("short interval test:")

    x = Xi(1234567, 9876543)
    print(x)
    for i in range(20):
        print(x(i))

    b = B_Xi(100, 1234567, 9876543)
    print(b)
    for i in range(20):
        print(b(i))

    print("sketch link_type test:")
    x = Xi(1675206430, 3737435780)
    b = B_Xi(50, 1664175982, 431896386)
    for i in range(1, 19):
        print("bucket {:>5}: {:>10}".format(b(i).item(), x(i).item()))
