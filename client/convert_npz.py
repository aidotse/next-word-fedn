from model import  convert_pth_to_npz

if __name__ == "__main__":
    model = convert_pth_to_npz("bert_weights.pth", "bert.npz")
