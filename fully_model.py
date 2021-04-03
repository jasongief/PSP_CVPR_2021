import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import pdb

def init_layers(layers):
    for layer in layers:
        nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0)


class SelfAttention(nn.Module):
    # Take audio self-attention for example.
    def __init__(self, audio_emb_dim, hidden_dim=64):
        super(SelfAttention, self).__init__()

        self.phi = nn.Linear(audio_emb_dim, hidden_dim)
        self.theta = nn.Linear(audio_emb_dim, hidden_dim)
        self.g = nn.Linear(audio_emb_dim, hidden_dim)
        layers = [self.phi, self.theta, self.g]
        init_layers(layers)

    def forward(self, audio_feature):
        # audio_feature: [bs, seg_num=10, 128]
        bs, seg_num, audio_emb_dim = audio_feature.shape
        phi_a = self.phi(audio_feature)
        theta_a = self.theta(audio_feature)
        g_a = self.g(audio_feature)
        a_seg_rel = torch.bmm(phi_a, theta_a.permute(0, 2, 1)) # [bs, seg_num, seg_num]
        a_seg_rel = a_seg_rel / torch.sqrt(torch.FloatTensor([audio_emb_dim]).cuda())
        a_seg_rel = F.relu(a_seg_rel)
        a_seg_rel = (a_seg_rel + a_seg_rel.permute(0, 2, 1)) / 2
        sum_a_seg_rel = torch.sum(a_seg_rel, dim=-1, keepdim=True)
        a_seg_rel = a_seg_rel / (sum_a_seg_rel + 1e-8)
        a_att = torch.bmm(a_seg_rel, g_a)
        a_att_plus_ori = a_att + audio_feature
        return a_att_plus_ori, a_seg_rel


class AVGA(nn.Module):
    """Audio-guided visual attention used in AVEL.
    AVEL:Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chen-liang Xu. Audio-visual event localization in unconstrained videos. InECCV, 2018
    """
    def __init__(self, a_dim=128, v_dim=512, hidden_size=512, map_size=49):
        super(AVGA, self).__init__()
        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(a_dim, hidden_size)
        self.affine_video = nn.Linear(v_dim, hidden_size)
        self.affine_v = nn.Linear(hidden_size, map_size, bias=False)
        self.affine_g = nn.Linear(hidden_size, map_size, bias=False)
        self.affine_h = nn.Linear(map_size, 1, bias=False)

        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)
        init.xavier_uniform(self.affine_audio.weight)
        init.xavier_uniform(self.affine_video.weight)

    def forward(self, audio, video):
        # audio: [bs, 10, 128]
        # video: [bs, 10, 7, 7, 512]
        V_DIM = video.size(-1)
        v_t = video.view(video.size(0) * video.size(1), -1, V_DIM) # [bs*10, 49, 512]
        V = v_t

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t)) # [bs*10, 49, 512]
        a_t = audio.view(-1, audio.size(-1)) # [bs*10, 128]
        a_t = self.relu(self.affine_audio(a_t)) # [bs*10, 512]
        content_v = self.affine_v(v_t) + self.affine_g(a_t).unsqueeze(2) # [bs*10, 49, 49] + [bs*10, 49, 1]

        z_t = self.affine_h((F.tanh(content_v))).squeeze(2) # [bs*10, 49]
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1)) # attention map, [bs*10, 1, 49]
        c_t = torch.bmm(alpha_t, V).view(-1, V_DIM) # [bs*10, 1, 512]
        video_t = c_t.view(video.size(0), -1, V_DIM) # attended visual features, [bs, 10, 512]
        return video_t


class LSTM_A_V(nn.Module):
    def __init__(self, a_dim, v_dim, hidden_dim=128, seg_num=10):
        super(LSTM_A_V, self).__init__()

        self.lstm_audio = nn.LSTM(a_dim, hidden_dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
        self.lstm_video = nn.LSTM(v_dim, hidden_dim, 1, batch_first=True, bidirectional=True, dropout=0.0)

    def init_hidden(self, a_fea, v_fea):
        bs, seg_num, a_dim = a_fea.shape
        hidden_a = (torch.zeros(2, bs, a_dim).cuda(), torch.zeros(2, bs, a_dim).cuda())
        hidden_v = (torch.zeros(2, bs, a_dim).cuda(), torch.zeros(2, bs, a_dim).cuda())
        return hidden_a, hidden_v

    def forward(self, a_fea, v_fea):
        # a_fea, v_fea: [bs, 10, 128]
        hidden_a, hidden_v = self.init_hidden(a_fea, v_fea)
        # Bi-LSTM for temporal modeling
        self.lstm_video.flatten_parameters() # .contiguous()
        self.lstm_audio.flatten_parameters()
        lstm_audio, hidden1 = self.lstm_audio(a_fea, hidden_a)
        lstm_video, hidden2 = self.lstm_video(v_fea, hidden_v)

        return lstm_audio, lstm_video


class PSP(nn.Module):
    """Postive Sample Propagation module"""

    def __init__(self, a_dim=256, v_dim=256, hidden_dim=256, out_dim=256):
        super(PSP, self).__init__()
        self.v_L1 = nn.Linear(v_dim, hidden_dim, bias=False)
        self.v_L2 = nn.Linear(v_dim, hidden_dim, bias=False)
        self.v_fc = nn.Linear(v_dim, out_dim, bias=False)
        self.a_L1 = nn.Linear(a_dim, hidden_dim, bias=False)
        self.a_L2 = nn.Linear(a_dim, hidden_dim, bias=False)
        self.a_fc = nn.Linear(a_dim, out_dim, bias=False)
        self.activation = nn.ReLU()
        # self.activation = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1) # default=0.1
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-6)

        layers = [self.v_L1, self.v_L2, self.a_L1, self.a_L2, self.a_fc, self.v_fc]
        self.init_weights(layers)

    def init_weights(self, layers):
        for layer in layers:
            nn.init.xavier_uniform(layer.weight)
            # nn.init.orthogonal(layer.weight)
            # nn.init.kaiming_normal_(layer.weight, mode='fan_in')

    def forward(self, a_fea, v_fea, thr_val):
        # a_fea: [bs, 10, 256]
        # v_fea: [bs, 10, 256]
        # thr_val: the hyper-parameter for pruing process
        v_branch1 = self.dropout(self.activation(self.v_L1(v_fea))) #[bs, 10, hidden_dim]
        v_branch2 = self.dropout(self.activation(self.v_L2(v_fea)))
        a_branch1 = self.dropout(self.activation(self.a_L1(a_fea)))
        a_branch2 = self.dropout(self.activation(self.a_L2(a_fea)))

        cross_att_v_to_a = torch.bmm(v_branch2, a_branch1.permute(0, 2, 1)) # row(v) - col(a), [bs, 10, 10]
        cross_att_v_to_a /= torch.sqrt(torch.FloatTensor([v_branch2.shape[2]]).cuda())

        cross_att_v_to_a = F.relu(cross_att_v_to_a) # ReLU
        cross_att_a_to_v = cross_att_v_to_a.permute(0, 2, 1) # transpose

        sum_v_to_a = torch.sum(cross_att_v_to_a, dim=-1, keepdim=True)
        cross_att_v_to_a = cross_att_v_to_a / (sum_v_to_a + 1e-8) # [bs, 10, 10]
        cross_att_v_to_a = (cross_att_v_to_a > thr_val).float() * cross_att_v_to_a
        sum_v_to_a = torch.sum(cross_att_v_to_a, dim=-1, keepdim=True)  # l1-normalization
        cross_att_v_to_a = cross_att_v_to_a / (sum_v_to_a + 1e-8)

        sum_a_to_v = torch.sum(cross_att_a_to_v, dim=-1, keepdim=True)
        cross_att_a_to_v = cross_att_a_to_v / (sum_a_to_v + 1e-8)
        cross_att_a_to_v = (cross_att_a_to_v > thr_val).float() * cross_att_a_to_v
        sum_a_to_v = torch.sum(cross_att_a_to_v, dim=-1, keepdim=True)
        cross_att_a_to_v = cross_att_a_to_v / (sum_a_to_v + 1e-8)

        a_pos = torch.bmm(cross_att_v_to_a, a_branch2)
        v_psp = v_fea + a_pos
        v_pos = torch.bmm(cross_att_a_to_v, v_branch1)
        a_psp = a_fea + v_pos

        v_psp = self.dropout(self.relu(self.v_fc(v_psp)))
        a_psp = self.dropout(self.relu(self.a_fc(a_psp)))
        v_psp = self.layer_norm(v_psp)
        a_psp = self.layer_norm(a_psp)

        a_v_fuse = torch.mul(v_psp + a_psp, 0.5)
        return a_v_fuse, v_psp, a_psp



class Classify(nn.Module):
    def __init__(self, hidden_dim=256, category_num=28):
        super(Classify, self).__init__()
        self.L1 = nn.Linear(hidden_dim, 64, bias=False)
        self.L2 = nn.Linear(64, category_num, bias=False)
        nn.init.xavier_uniform(self.L1.weight)
        nn.init.xavier_uniform(self.L2.weight)
    def forward(self, feature):
        out = F.relu(self.L1(feature))
        out = self.L2(out)
        # out = F.softmax(out, dim=-1)
        return out


class AVSimilarity(nn.Module):
    """ function to compute audio-visual similarity"""
    def __init__(self,):
        super(AVSimilarity, self).__init__()

    def forward(self, v_fea, a_fea):
        # fea: [bs, 10, 256]
        v_fea = F.normalize(v_fea, dim=-1)
        a_fea = F.normalize(a_fea, dim=-1)
        cos_simm = torch.sum(torch.mul(v_fea, a_fea), dim=-1) # [bs, 10]
        return cos_simm



class psp_net(nn.Module):
    '''
    System flow for fully supervised audio-visual event localization.
    '''
    def __init__(self, a_dim=128, v_dim=512, hidden_dim=128, category_num=29):
        super(psp_net, self).__init__()
        self.fa = nn.Sequential(
            nn.Linear(a_dim, 256, bias=False),
            nn.Linear(256, 128, bias=False),
        )
        self.fv = nn.Sequential(
            nn.Linear(v_dim, 256, bias=False),
            nn.Linear(256, 128, bias=False),
        )
        self.linear_v = nn.Linear(v_dim, a_dim)
        self.relu = nn.ReLU()
        self.attention = AVGA(v_dim=v_dim)
        self.lstm_a_v = LSTM_A_V(a_dim=a_dim, v_dim=hidden_dim, hidden_dim=hidden_dim)
        self.psp = PSP(a_dim=a_dim*2, v_dim=hidden_dim*2)
        self.av_simm = AVSimilarity()

        self.v_classify = Classify(hidden_dim=256)
        self.a_classify = Classify(hidden_dim=256)

        self.L1 = nn.Linear(2*hidden_dim, 64, bias=False)
        self.L2 = nn.Linear(64, category_num, bias=False)

        self.L3 = nn.Linear(256, 64)
        self.L4 = nn.Linear(64, 2)
        # layers = [self.L1, self.L2]
        layers = [self.L1, self.L2, self.L3, self.L4]
        self.init_layers(layers)

    def init_layers(self, layers):
        for layer in layers:
            nn.init.xavier_uniform(layer.weight)

    def forward(self, audio, video, thr_val):
        # audio: [bs, 10, 128]
        # video: [bs, 10, 7, 7, 512]
        bs, seg_num, H, W, v_dim = video.shape
        fa_fea = self.fa(audio)
        video_t = self.attention(fa_fea, video) # [bs, 10, 512]
        video_t = self.fv(video_t) # [bs, 10, 128]
        lstm_audio, lstm_video = self.lstm_a_v(fa_fea, video_t)
        fusion, final_v_fea, final_a_fea = self.psp(lstm_audio, lstm_video, thr_val=thr_val) # [bs, 10, 256]
        cross_att = self.av_simm(final_v_fea, final_a_fea)

        out = self.relu(self.L1(fusion))
        out = self.L2(out) # [bs, 10, 29]
        return fusion, out, cross_att

