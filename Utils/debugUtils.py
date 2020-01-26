import time

def measure_time():
    def wraps(func):
        def mesure(*args,**kwargs):
            start = time.time()
            res = func(*args,**kwargs)
            end = time.time()
            # logger.info("function %s use time %s"%(func.__name__,(end-start)))
            print("function %s use time %s"%(func.__name__,(end-start)))
            return res
        return mesure
    return wraps

def itemmap():
    def wraps(func):
        def changeinx(s, inx):
            inx_n = inx
            i = inx % 317
            if i < 50 or 160 < i < 210:
                inx_n -= 50
            if 50 < i <110 or 210 < i < 270:
                inx_n += 50
            if inx_n >= 5000:
                inx_n -= 317
            if inx_n < 0:
                inx_n += 317
            res = func(s, inx_n)
            return res
        return changeinx
    return wraps