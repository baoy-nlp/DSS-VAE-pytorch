import torch.nn as nn


def reflect(x):
    return x


class Bridge(nn.Module):
    def __init__(self, rnn_type, mapper_type, encoder_dim, encoder_layer, decoder_dim, decoder_layer):
        super(Bridge, self).__init__()
        self.rnn_type = rnn_type
        self.parameter_type = mapper_type
        self.input_dim = encoder_dim * encoder_layer
        self.output_dim = decoder_dim * decoder_layer
        self.decoder_dim = decoder_dim
        self.decoder_layer = decoder_layer

        if self.parameter_type == "mapping":
            self.mapper = nn.Linear(in_features=self.input_dim, out_features=self.output_dim)
        elif self.input_dim == self.output_dim:
            self.mapper = reflect

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: [layers,batch,direction * hidden_size]
        """
        batch_size = input_tensor.size(1)
        reset_tensor = input_tensor.permute(1, 0, 2).contiguous()
        reset_tensor = reset_tensor.view(batch_size, -1)
        assert reset_tensor.size(1) == self.input_dim, "bridge dim is not right"
        output_tensor = self.mapper(reset_tensor).view(-1, self.decoder_layer, self.decoder_dim)
        return output_tensor.permute(1, 0, 2).contiguous()
