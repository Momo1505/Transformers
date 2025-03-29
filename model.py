import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self,d_model : int,seq_len : int,dropout : float):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #create the positional_encoding matrix of shape (seq_len,d_model)
        pe = torch.zeros(seq_len,d_model,dtype=torch.float)
        #compute the numerator of shape (seq_len,1)
        numerator = torch.arange(seq_len,dtype=torch.float).unsqueeze(1)
        #create the denominator
        denominator = torch.exp(torch.arange(0,d_model,2).float() * (- math.log(10000.0) / d_model))
        #apply the sin to even positions
        pe[:,0::2] = torch.sin(numerator*denominator)
        #apply the cos to the odd position
        pe[:,1::2] = torch.cos(numerator*denominator)

        pe = pe.unsqueeze(0) # shape (1,seq_len,d_model)
        self.register_buffer("pe",pe)

    def forward(self,x):
        x = x + (self.pe[:,:x.size(1),:]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self,eps:float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1)) # multiplied
        self.beta = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        means = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return ((x - means) * self.gamma) / (std + self.eps) + self.beta

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, dff:int, dropout:float):
        super(FeedForwardBlock, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model,dff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff,d_model)
        )

    def forward(self,x):
        # (batch,seq_len,d_model) --> (batch,seq_len,dff) --> (batch,seq_len,d_model)
        return self.feed_forward(x)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int, h:int, dropout:float):
        super(MultiHeadAttentionBlock,self).__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def attention(query,key,value,mask,dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch,h,seq_len,d_k) --> (batch,h,seq_len,seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, 1e-9)

        attention_scores = attention_scores.softmax(dim=-1) # (batch,h,seq_len,seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self,q,k,v,mask):
        query = self.w_q(q) # (batch,seq_len,d_model) --> (batch,seq_len,d_model)
        key = self.w_k(k) # (batch,seq_len,d_model) --> (batch,seq_len,d_model)
        value = self.w_v(v) # (batch,seq_len,d_model) --> (batch,seq_len,d_model)

        # (batch,seq_len,d_model) --> (batch,seq_len,h,d_k) -> (batch,h,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # (batch,h,seq_len,d_k) --> (batch,seq_len,h,d_k) --> (batch,seq_len,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)

        # (batch,seq_len,d_model) --> (batch,seq_len,d_model) 
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout:float):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super(EncoderBlock,self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x,self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self,layers : nn.ModuleList):
        super(Encoder,self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,mask):
        for encoder_block in self.layers:
            x = encoder_block(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block : MultiHeadAttentionBlock,
                 cross_attention_block : MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.residual_connections[0](x,lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x,lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x,self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super(Decoder,self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for decoder_block in self.layers:
            x = decoder_block(x,encoder_output,src_mask,tgt_mask)

        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int, vocab_size:int):
        super(ProjectionLayer,self).__init__()
        self.projection = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        # (batch,seq_len,d_model) --> # (batch,seq_len,vocab_size)
        return torch.log_softmax(self.projection(x), dim=-1 )

class Transformer(nn.Module):
    def __init__(self,encoder: Encoder,
                 decoder: Decoder,
                 input_embeddings: InputEmbeddings,
                 target_embeddings: InputEmbeddings,
                 input_position: PositionalEncoding,
                 target_position: PositionalEncoding,
                 projection: ProjectionLayer):
        super(Transformer,self).__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.input_embeddings = input_embeddings
        self.target_embeddings = target_embeddings
        self.input_position = input_position
        self.target_position = target_position
        self.projection = projection

    def encode(self,input, input_mask):
        x = self.input_embeddings(input)
        x = self.input_position(x)
        return self.encoder(x,input_mask)

    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt = self.target_embeddings(tgt)
        tgt = self.target_position(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        return self.projection(x)

def build_transformer(
        src_vocab_size : int,
        tgt_vocab_size : int,
        src_seq_len : int,
        tgt_seq_len: int,
        d_model : int = 512,
        N : int = 6, # number of block
        h : int = 8,
        dropout : float = 0.1,
        dff : int = 2048
) -> Transformer:
    # create the embedding layer
    src_embedding = InputEmbeddings(d_model,src_vocab_size)
    tgt_embedding = InputEmbeddings(d_model,tgt_vocab_size)

    # create the positional encoding
    src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model,tgt_seq_len,dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,encoder_feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,decoder_feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    # create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create projection
    projection = ProjectionLayer(d_model,tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection)

    # initialize the model parameters
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_normal_(param)

    return transformer