import torch.nn as nn

import torch
import torch.nn.functional as F
import torch.optim as optim

import os

class NoamScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = max(1, self._step_count)
        scale = (self.d_model ** (-0.5)) * min(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5))
        update_lr = [base_lr * scale for base_lr in self.base_lrs]
        #update_lr = torch.tensor(update_lr).to(self.device)
        return update_lr

class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, maxpool=False, maxpool_k=2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.maxpool = maxpool
        self.maxpool_k = maxpool_k

        self.conv1d = nn.Conv1d(self.in_channels, self.out_channels, \
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.leaky_relu = nn.LeakyReLU()
        self.maxpool = None
        # pytorch default for stride is now 2, will use stride 2 to see what happens with result
        # may need to change
        if maxpool:
            self.maxpool = nn.MaxPool1d(kernel_size=self.maxpool_k, stride=2)
        
    def forward(self, input):
        self.batch, self.in_channel, self.seq_len = input.shape[0], input.shape[1], input.shape[2]

        output = self.conv1d(input)
        #print(f"output shape in cnn block: {output.shape}")
        output = self.leaky_relu(output)
        
        if self.maxpool != None:
            output = self.maxpool(output)
        
        return output


class OneDCnnEncoder(nn.Module):
    def __init__(self, cnn_config):
        super().__init__()

        self.n_conv = cnn_config["n_conv"]
        self.n_filters = cnn_config["n_filters"]
        self.d_model = cnn_config["d_model"] # dimension of the model
        self.out_channels = self.d_model
        self.d_proj = cnn_config["d_proj"]
        self.in_channels = cnn_config["in_channels"]

        # first apply the initial filters of 32 to create 32 channels
        # need to double check if these are the input filters of data.

        # (187 + 2(0) - 3) // 1 + 1 = 185
        self.conv_block1 = CnnBlock(self.in_channels, self.out_channels)

        # (184 + 2(1) - 3) // 1 + 1 = 185
        self.conv_block2 = CnnBlock(self.out_channels, self.out_channels, padding=1)

        # conv 1
        # (185 + 2(0) - 3) // 1 + 1 = 183  
        # maxpool 1
        # the pytorch versin running have a new dimension calculation
        # out_channel = (in channel + 2 * padding - [dilation (1) * (kernel size - 1)] - 1) // (2) stride + 1
        # (183 - (2-1) - 1 ) // 2 + 1 = 91
        # conv 2
        # (91 + 2(0) - 3) // 1 + 1 = 89
        # maxpool 2
        # (89 - (2-1) - 1) // 2 + 1 = 44
        # conv 3
        # (44 + 2(0) - 3) // 1 + 1 = 42
        # max pool 3
        # (42 - (2-1) - 1) // 2 + 1 = 21
        # conv 4
        # (21 + 2(0) - 3) // 1 + 1 = 19
        # max pool 3
        # (19 - (2-1) - 1) // 2 + 1 = 9
        self.conv_block3 = nn.ModuleList([
            CnnBlock(self.out_channels, self.out_channels, maxpool=True) for i in range(self.n_conv)
        ])
    
    def forward(self, input):

        #print(f"input shape {input.shape}")
        #output = self.conv_block1(input)
        #output = self.conv_block2(output)

        output = self.conv_block1(input)
        output = self.conv_block2(output)
        for i in range(self.n_conv):
            #print(f"convolution: {i}")
            #print(f"output shape before forward: {output.shape}")
            output = self.conv_block3[i].forward(output)
            #print(f"output of block device: {output.device}")
            #print(f"output shape after block forward {output.shape}")
        
        # need to permute to feed into the embeddings
        #print(f"final output shape in OneDCnnEncoder: {output.shape}")
        return output

class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, d_model=128, max_len=187):

        super().__init__()

        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # PE even sin(pos/(10000 * pow(2*(i/d))))
        self.max_len = max_len
        self.positional_embeddings = torch.zeros(self.max_len, d_model).to(self.device)
        # 1 x max positions
        self.position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1).to(self.device)
        # create the position i multiplied by 2 by creating the range in increments of 2
        # in this case when take the power of - it is taking it as the denominator term
        self.division_term = torch.pow(10000, -torch.arange(0, d_model, 2).float() / d_model).to(self.device)
        # even terms do sine
        self.positional_embeddings[:, 0::2] = torch.sin(self.position * self.division_term)
        # odd terms do cosine
        self.positional_embeddings[:, 1::2] = torch.cos(self.position * self.division_term)

        self.register_buffer('pe', self.positional_embeddings.unsqueeze(1))
    
    def forward(self, embeddings):
        # input dimension is batch_size x seq_len x d_model = 128
        seq_len = embeddings.shape[1]
        # need to add a dimension for batch in order to change it to 1 x seq len(1) x dmodel 
        # for broadcasting add
        embeddings = embeddings.to(self.device)
        output = embeddings + self.positional_embeddings[:seq_len].unsqueeze(0)

        return output

        
class TransformerEncoder(nn.Module):
    def __init__(self, transformer_config):
        super().__init__()

        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.n_head = transformer_config["n_head"]
        self.n_layers = transformer_config["n_layers"]
        self.d_hid = transformer_config["d_hid"]
        self.dropout = transformer_config["dropout"]
        self.embedding_dim = transformer_config["embedding_dim"]
        self.num_classes = transformer_config["num_classes"]

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, \
            nhead=self.n_head, dim_feedforward=self.d_hid, dropout=self.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)
        self.positional_embeddings = SinusoidalPositionalEmbeddings()
        self.layer_norm1 = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim, self.num_classes)


    def forward(self, embeddings):

        # create the positional encoding
        embeddings = embeddings.to(self.device)
        final_embeddings = self.positional_embeddings(embeddings)
        encoder_layer_output = self.encoder(final_embeddings)

        H_TE = encoder_layer_output # Feature representation needed for fusion layer

        # extract the cls variable
        #print(f"encoder layer output shape: {encoder_layer_output.shape}")
        # batch_size x cls x d_model
        cls_entry = encoder_layer_output[:, 0, :].to(self.device)
        cls_entry = self.layer_norm1(cls_entry)
        #print(f"cls_entry output shape: {cls_entry.shape}")
        output = self.fc1(cls_entry)
        #print(f"first linear layer output {output.shape}")
        output = self.layer_norm2(output)
        final_output = self.fc2(output)

        return final_output, H_TE


class OneDCnnTransformer(nn.Module):
    def __init__(self, cnn_config, transformer_config, batch_size):
        super().__init__()

        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.cnn_config = cnn_config
        self.transformer_config = transformer_config
        self.batch_size = batch_size
        self.embedding_dim = transformer_config["embedding_dim"]

        self.one_d_cnn_encoder = OneDCnnEncoder(self.cnn_config)
        
        self.one_d_transformer_encoder = TransformerEncoder(transformer_config)

    def forward(self, input):

        #print(f"input shape in OneDCnnTransformer: {input.shape}")
        input = input.unsqueeze(1)
        output = self.one_d_cnn_encoder(input)
        # need to permute the dimensions for embeddings
        # batch_size x seq len x 128 (embedding dim)
        output = output.permute(0, 2, 1)
        #print(f"output shape before embedding: {output.shape}")
    
        cls_zero = torch.zeros(1, 1, self.embedding_dim).to(self.device)
        cls_token = nn.Parameter(cls_zero)
        cls_tokens = cls_token.expand(input.shape[0], -1, -1)  # batchsize, 1, embedding_dim
        embeddings_with_cls = torch.cat((cls_tokens, output), dim=1) # batch_size, seq_len + 1, embedding_dim

        output, H_TE = self.one_d_transformer_encoder(embeddings_with_cls)
        #print(f"output shape after transformer encoder: {output.shape}")

        #print(f"Encoder output: {output.shape}")
        #print(f"output first entry: {output[0]}")
        #scores = torch.softmax(output, dim=1)
        #print(f"scores: {scores[0]}")

        return output, H_TE
