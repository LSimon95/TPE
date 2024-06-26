from ctypes import cdll, c_char_p, c_int, POINTER

class TPE(object):
    def __init__(self, libtpe_path : str, vocab_path : str):
        self.lib = cdll.LoadLibrary(libtpe_path)

        self.lib.tpe_new.restype = POINTER(c_int)

        self.obj = self.lib.tpe_new(c_char_p(vocab_path.encode("utf-8")))

        self.lib.free_ptr.argtypes = (POINTER(c_int),)
        self.lib.free_ptr.restype = None

        self.lib.tpe_st2at.argtypes = (POINTER(c_int), POINTER(c_int), c_int, c_int)
        self.lib.tpe_st2at.restype = POINTER(c_int)

        self.lib.tpe_at2st.argtypes = (POINTER(c_int), POINTER(c_int), c_int, c_int)
        self.lib.tpe_at2st.restype = POINTER(c_int)

    def __del__(self):
        self.lib.tpe_delete(self.obj)

    def st2at(self, st : list[int], max_len : int = 8192):
        if len(st) == 0:
            return []

        st = (c_int * len(st))(*st)
        at = self.lib.tpe_st2at(self.obj, st, len(st), 8192)

        # copy the result to a list
        at_list = []
        for i in range(at[0]):
            at_list.append(at[i+1])

        self.lib.free_ptr(at)
        return at_list

    def at2st(self, at : list[int], max_len : int = 16384):
        if len(at) == 0:
            return []

        at = (c_int * len(at))(*at)
        st = self.lib.tpe_at2st(self.obj, at, len(at), max_len)

        # copy the result to a list
        st_list = []
        for i in range(st[0]):
            st_list.append(st[i+1])
        
        self.lib.free_ptr(st)
        return st_list


if __name__ == "__main__":

    import random
    import time
    tpe = TPE("/home/qanpan.lo/ds1/TPE/libtpe.so", "/home/qanpan.lo/52271.txt")

    while True:

        stfake = [random.randint(0, 64000 + 52271 - 1) for _ in range(random.randint(0, 4599))]
        at = tpe.st2at(stfake)

        # print(at)

        start_time = time.time()
        st = tpe.at2st(at)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("--- ST Token length: %d ---" % len(st))


        start_time = time.time()
        at_decode = tpe.st2at(st)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("--- AT Token length: %d ---" % len(at_decode))

        # print(at)
        # print(st)

        if at != at_decode:
            print("Error")
            print(at)
            print(st)
            print(at_decode)
            break


