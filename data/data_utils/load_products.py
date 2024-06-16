import torch
import pandas as pd

FILE = './datasets/ogbn_products/ogbn_products_orig/ogbn-products.csv'


def get_raw_text_products(use_text=False, seed=0):
    data = torch.load('./datasets/ogbn_products/ogbn_products/ogbn-products_subset.pt')
    text = pd.read_csv('./datasets/ogbn_products/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric()
    data.y = data.y.squeeze()

    if not use_text:
        return data, None

    return data, text


if __name__ == '__main__':
    data, text = get_raw_text_products(True)
    print(data)
    print(text[0])
    