import torch
import threading
from cuda import cuda

def print_current_ctx(i: int):
    err, ctx = cuda.cuCtxGetCurrent()
    print(f"{i}, {ctx}")

# Let's start Pytorch first
    torch.cuda.init()

def thread(i: int):

    print_current_ctx(i)
    torch.zeros((1, 1), device='cuda')
    print_current_ctx(i)

    # Change the CUDA context afterwards?
    err, device = cuda.cuDeviceGet(0)
    err, cuda_context = cuda.cuCtxCreate(0, device)
    err, = cuda.cuCtxPushCurrent(cuda_context) #push my context back on the stack, since other programs need not be aware of my program
    print_current_ctx(i)

    # What context is this using?
    zeros = torch.zeros((10000, 10000), device="cuda")
    print_current_ctx(i)
    del zeros

    err, cuda_context = cuda.cuCtxPopCurrent() 
    cuda.cuCtxDestroy(cuda_context)

t0 = threading.Thread(target=thread, args=(0,))
t1 = threading.Thread(target=thread, args=(1,))
t0.start()
t1.start()

t0.join()
t1.join()