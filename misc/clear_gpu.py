import torch
import gc

def main():
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()

