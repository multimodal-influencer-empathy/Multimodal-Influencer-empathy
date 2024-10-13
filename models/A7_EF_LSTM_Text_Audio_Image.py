import torch
import torch.nn as nn
import torch.nn.functional as F

class A7_EF_LSTM_Text_Audio_Image(nn.Module):
    """
    early fusion using lstm
    """
    def __init__(self, args):
        super(A7_EF_LSTM_Text_Audio_Image, self).__init__()
        text_in, audio_in, video_in = args.feature_dims

        in_size = text_in + audio_in + video_in
        input_len =  args.seq_lens


        hidden_size = args.hidden_dims
        num_layers = args.num_layers
        dropout = args.dropout
        output_dim =  1


        self.norm = nn.BatchNorm1d(input_len)
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True, batch_first=True) # bidirectional
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_dim)



    def forward(self, text_x, audio_x, video_x):


        x = torch.cat([text_x, audio_x, video_x], dim=-1)

        x = self.norm(x)
        _, final_states = self.lstm(x)
        x = self.dropout(final_states[0][-1].squeeze())
        x = F.relu(self.linear(x), inplace=True)
        x = self.dropout(x)
        output = self.out(x)
        res = {
            'M': output
        }
        return res 
