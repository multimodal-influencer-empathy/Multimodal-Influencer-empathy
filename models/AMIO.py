import torch.nn as nn
from models.subNets import AlignSubNet
from models.A1_LSTM_Text import A1_LSTM_Text
from models.A2_LSTM_Audio import A2_LSTM_Audio
from models.A3_LSTM_Image import A3_LSTM_Image
from models.A4_EF_LSTM_Text_Audio import A4_EF_LSTM_Text_Audio
from models.A5_EF_LSTM_Text_Image import A5_EF_LSTM_Text_Image
from models.A6_EF_LSTM_Audio_Image import A6_EF_LSTM_Audio_Image
from models.A7_EF_LSTM_Text_Audio_Image import A7_EF_LSTM_Text_Audio_Image
from models.A8_TFN import A8_TFN
from models.A9_LMF import A9_LMF
from models.A10_MFN import A10_MFN
from models.A11_Graph_MFN_noG import A11_Graph_MFN_noG
from models.A12_Graph_MFN_noW import A12_Graph_MFN_noW
from models.A13_Graph_MFN_noM import A13_Graph_MFN_noM
from models.A14_Graph_MFN import A14_Graph_MFN

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.MODEL_MAP = {
            'A1_lstm_text': A1_LSTM_Text,
            'A2_lstm_audio': A2_LSTM_Audio,
            'A3_lstm_image': A3_LSTM_Image,
            'A4_ef_lstm_text_audio': A4_EF_LSTM_Text_Audio,
            'A5_ef_lstm_text_image': A5_EF_LSTM_Text_Image,
            'A6_ef_lstm_audio_image': A6_EF_LSTM_Audio_Image,
            'A7_ef_lstm_text_audio_image': A7_EF_LSTM_Text_Audio_Image,
            'A8_tfn': A8_TFN,
            'A9_lmf': A9_LMF,
            'A10_mfn': A10_MFN,
            'A11_graph_mfn_noG': A11_Graph_MFN_noG,
            'A12_graph_mfn_noW': A12_Graph_MFN_noW,
            'A13_graph_mfn_noM': A13_Graph_MFN_noM,
            'A14_graph_mfn': A14_Graph_MFN
        }
        self.need_model_aligned = args['need_model_aligned']
        if(self.need_model_aligned):
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()
        lastModel = self.MODEL_MAP[args['model_name']]
        print("Model:", args['model_name'])
        print("\n The args in the model is as below: \n", args)
        self.model_name = args['model_name']
        self.Model = lastModel(args)


    def forward(self, text_x, audio_x, vision_x, *args,**kwargs):
        if (self.need_model_aligned):
            text_x, audio_x, vision_x = self.alignNet(text_x, audio_x, vision_x)
        return self.Model(text_x, audio_x, vision_x,  *args, **kwargs)
