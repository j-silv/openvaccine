import torch
from openvaccine.model import mask_tokens

def test_mask_tokens():
    tokens = torch.arange(10).reshape(2, 5)
    mask_token = 10 # 11th token as mask
    vocab_length = 10 # 10 tokens as valid

    # with this seed and setup i precomputed the expected selected tokens
    # and the masked results
    torch.manual_seed(26)
    masked_tokens_idx, masked_tokens = mask_tokens(tokens, mask_token, vocab_length, 
                                                   percent=0.8, mask_prob=0.33,
                                                   random_prob=0.33, same_prob=0.34)

    assert torch.all(masked_tokens_idx == torch.tensor([[False, True, False, True,  True],
                                                        [True,  True,  True, False, True]]))

    # 3 and 4 stays the same; 9 was selected randomly
    assert torch.all(masked_tokens == torch.tensor([[0, mask_token, 2, 3, 4],
                                                    [mask_token,  mask_token,  9, 8, mask_token]]))
    
    # test if no tokens are selected they are all the same as original
    torch.manual_seed(26)
    masked_tokens_idx, masked_tokens = mask_tokens(tokens, mask_token, vocab_length, 
                                                   percent=0, mask_prob=0.33,
                                                   random_prob=0.33, same_prob=0.34)
    
    assert torch.all(masked_tokens == tokens)


    # test if all tokens are masked
    torch.manual_seed(26)
    masked_tokens_idx, masked_tokens = mask_tokens(tokens, mask_token, vocab_length, 
                                                   percent=1.0, mask_prob=1.0,
                                                   random_prob=0.0, same_prob=0.0)
    
    assert torch.all(masked_tokens == mask_token)