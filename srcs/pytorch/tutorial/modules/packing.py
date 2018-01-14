import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Helper functions
def pack(seq):
    '''
    Packs a list of variable-length tensors into a packed sequence
    
    Args:
        seq: 2 dim tensor, where each row corresponds to an individual element.
        
    Returns:
        packed: PackedSequence
        orders: ordered indices for the original sequence before the sorting.
        later used to retrieve the original ordering of the sequences.
        
    '''
    seq_sorted = []
    orders = []
    
    for i, tensor in sorted(enumerate(seq), key = lambda t: -t[1].size()[0]):
        seq_sorted.append(tensor)
        orders.append(i)
        
    lengths = list(map(lambda t: t.size()[0], seq_sorted))
    
    max_seq_len = lengths[0]
    dim = seq_sorted[0].size()[1]
    batch_size = len(seq_sorted)
    
    # Build a padded sequence
    padded_sequence = Variable(torch.zeros(max_seq_len, batch_size, dim))
    if torch.cuda.is_available():
        padded_sequence = padded_sequence.cuda()
    
    for i in range(batch_size):
        padded_sequence[:lengths[i], i, :] = seq_sorted[i]
    
    # pack the padded sequence
    packed = pack_padded_sequence(padded_sequence, lengths)
    
    return packed, orders

def unpack(packed, orders):
    '''
    Unpacks a packed sequence
    
    Args:
        packed: PackedSequence
        
    Returns:
        unpacked_masked
    '''
    unpacked, lengths = pad_packed_sequence(packed)
    
    # Masking
    unpacked_masked = [unpacked[:lengths[batch], batch, :] for batch in range(len(lengths))]
    
    # Unsort
    unpacked_masked = [tensor for i, tensor in sorted(zip(orders, unpacked_masked))]
        
    return unpacked_masked
