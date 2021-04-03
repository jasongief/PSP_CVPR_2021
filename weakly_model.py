import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
import pdb

def init_layers(layers):
    for layer in layers:
        nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0)


class AudioSelfAttention(nn.Module):
    def __init__(self, audio_emb_dim, hidden_dim=64):
        super(AudioSelfAttention, self).__init__()

        self.phi = nn.Linear(audio_emb_dim, hidden_dim)
        self.theta = nn.Linear(audio_emb_dim, hidden_dim)
        self.g = nn.Linear(audio_emb_dim, hidden_dim)
        layers = [self.phi, self.theta, self.g]
        init_layers(layers)


    def forward(self, audio_feature):
        # audio_feature: [bs, 10, 128]
        bs, seg_num, audio_emb_dim = audio_feature.shape
        phi_a = self.phi(audio_feature)
        theta_a = self.theta(audio_feature)
        g_a = self.g(audio_feature) # [bs, 10, 128]
        a_seg_rel = torch.bmm(phi_a, theta_a.permute(0, 2, 1)) # [bs, seg_num, seg_num]
        a_seg_rel = a_seg_rel / torch.sqrt(torch.FloatTensor([audio_emb_dim]).cuda())
        a_seg_rel = F.relu(a_seg_rel)
        a_seg_rel = (a_seg_rel + a_seg_rel.permute(0, 2, 1)) / 2 #
        sum_a_seg_rel = torch.sum(a_seg_rel, dim=-1, keepdim=True)
        a_seg_rel = a_seg_rel / (sum_a_seg_rel + 1e-8)
        a_seg_rel = a_seg_rel * (a_seg_rel > 0.099).float()

        a_att = torch.bmm(a_seg_rel, g_a)
        a_att_plus_ori = a_att + audio_feature
        return a_att_plus_ori, a_seg_rel


class LSTM_A_V(nn.Module):
    def __init__(self, a_dim, v_dim, hidden_dim=128, category_num=29, seg_num=10):
        super(LSTM_A_V, self).__init__()

        self.lstm_audio = nn.LSTM(a_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.lstm_video = nn.LSTM(v_dim, hidden_dim, 1, batch_first=True, bidirectional=True)

    def init_hidden(self, a_fea, v_fea):
        bs, seg_num, a_dim = a_fea.shape
        hidden_a = (torch.zeros(2, bs, a_dim).cuda(), torch.zeros(2, bs, a_dim).cuda())
        hidden_v = (torch.zeros(2, bs, a_dim).cuda(), torch.zeros(2, bs, a_dim).cuda())
        return hidden_a, hidden_v

    def forward(self, a_fea, v_fea):
        # a_fea, v_fea: [bs, 10, 128]
        hidden_a, hidden_v = self.init_hidden(a_fea, v_fea)
        self.lstm_video.flatten_parameters() # .contiguous()
        self.lstm_audio.flatten_parameters()
        lstm_audio, hidden1 = self.lstm_audio(a_fea, hidden_a)
        lstm_video, hidden2 = self.lstm_video(v_fea, hidden_v)

        return lstm_audio, lstm_video



class PSP(nn.Module):
    def __init__(self, a_dim=256, v_dim=256, hidden_dim=256, out_dim=256):
        super(PSP, self).__init__()
        self.v_L1 = nn.Linear(v_dim, hidden_dim, bias=False)
        self.v_L2 = nn.Linear(v_dim, hidden_dim, bias=False)
        self.a_L1 = nn.Linear(a_dim, hidden_dim, bias=False)
        self.a_L2 = nn.Linear(a_dim, hidden_dim, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-6)
        layers = [self.v_L1, self.v_L2, self.a_L1, self.a_L2]
        self.init_weights(layers)

    def init_weights(self, layers):
        for layer in layers:
            nn.init.xavier_uniform(layer.weight)

    def forward(self, a_fea, v_fea, thr_val):
        # a_fea: [bs, 10, 256]
        # v_fea: [bs, 10, 256]
        v_branch1 = self.dropout(self.activation(self.v_L1(v_fea))) #[bs, 10, hidden_dim]
        v_branch2 = self.dropout(self.activation(self.v_L2(v_fea)))
        a_branch1 = self.dropout(self.activation(self.a_L1(a_fea)))
        a_branch2 = self.dropout(self.activation(self.a_L2(a_fea)))

        cross_att_v_to_a = torch.bmm(v_branch2, a_branch1.permute(0, 2, 1)) # row(v) - col(a), [bs, 10, 10]
        cross_att_v_to_a /= torch.sqrt(torch.FloatTensor([v_branch2.shape[2]]).cuda())

        cross_att_v_to_a = F.relu(cross_att_v_to_a) # relu
        cross_att_a_to_v = cross_att_v_to_a.permute(0, 2, 1) # transpose
        sum_v_to_a = torch.sum(cross_att_v_to_a, dim=-1, keepdim=True)
        cross_att_v_to_a = cross_att_v_to_a / (sum_v_to_a + 1e-8)
        cross_att_v_to_a = (cross_att_v_to_a > thr_val).float() * cross_att_v_to_a
        sum_a_to_v = torch.sum(cross_att_a_to_v, dim=-1, keepdim=True)
        cross_att_a_to_v = cross_att_a_to_v / (sum_a_to_v + 1e-8)
        cross_att_a_to_v = (cross_att_a_to_v > thr_val).float() * cross_att_a_to_v

        cross_v_fea = torch.bmm(cross_att_v_to_a, a_branch2)
        final_v_fea = v_fea + cross_v_fea
        cross_a_fea = torch.bmm(cross_att_a_to_v, v_branch1)
        final_a_fea = a_fea + cross_a_fea

        final_v_fea = self.layer_norm(final_v_fea)
        final_a_fea = self.layer_norm(final_a_fea)

        a_v_fuse = torch.mul(final_v_fea + final_a_fea, 0.5)
        return a_v_fuse, final_v_fea, final_a_fea, cross_v_fea, cross_a_fea, cross_att_v_to_a, cross_att_a_to_v


class AVGA(nn.Module):
    def __init__(self, hidden_size=512):
        super(AVGA, self).__init__()
        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(128, hidden_size)
        self.affine_video = nn.Linear(512, hidden_size)
        self.affine_v = nn.Linear(hidden_size, 49, bias=False)
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)
        self.affine_h = nn.Linear(49, 1, bias=False)

        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)
        init.xavier_uniform(self.affine_audio.weight)
        init.xavier_uniform(self.affine_video.weight)

    def forward(self, audio, video):
        #  audio: [bs, 10, 128]
        # video: [bs, 10, 7, 7, 512]
        v_t = video.view(video.size(0) * video.size(1), -1, 512)
        V = v_t

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t))
        a_t = audio.view(-1, audio.size(-1))
        a_t = self.relu(self.affine_audio(a_t))
        content_v = self.affine_v(v_t) + self.affine_g(a_t).unsqueeze(2)

        z_t = self.affine_h((F.tanh(content_v))).squeeze(2)
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1)) # attention map
        c_t = torch.bmm(alpha_t, V).view(-1, 512)
        video_t = c_t.view(video.size(0), -1, 512) #attended visual features

        return video_t

class DotAttention(nn.Module):
    def __init__(self, a_dim=128, v_dim=512, hid_dim=256):
        super(DotAttention, self).__init__()
        self.linear_a = nn.Linear(a_dim, hid_dim)
        self.linear_v = nn.Linear(v_dim, hid_dim)
        layers = [self.linear_a, self.linear_v]

        init_layers(layers)

    def forward(self, audio, video):
        # audio: [bs, 10, 128]
        # video: [bs, 10, 7, 7, 512]
        bs, seg_num, H, W, C = video.shape
        a_fea = self.linear_a(audio) # [bs, 10, hid_dim]
        a_fea = a_fea.view(bs*seg_num, -1).unsqueeze(1) # [bs*10, 1, hid_dim]
        video_t = video.view(bs*seg_num, -1, C) #[bs*10, 49, 512]
        v_fea = self.linear_v(video_t) # [bs*10, 49, hid_dim]
        a_fea = F.normalize(a_fea, dim=-1)
        v_fea = F.normalize(v_fea, dim=-1)

        att_wei = torch.bmm(a_fea, v_fea.permute(0, 2, 1)) # [bs*10, 1, 49]
        att_wei = F.softmax(att_wei, dim=-1)
        # att_wei = att_wei.view(bs, seg_num, H*W) # [bs, 10, 49] after softmax
        att_v_fea = torch.bmm(att_wei, video_t).view(bs, seg_num, C) # [bs*10, 1, 512] -> [bs, 10, 512]

        return att_v_fea



class psp_net(nn.Module):
    '''
    weakly supervised AVE localization
    '''
    def __init__(self, a_dim=128, v_dim=512, hidden_dim=128, category_num=29, num_DualBlock=1):
        super(psp_net, self).__init__()

        self.temperature = nn.Parameter(torch.ones(1, category_num))
        self.num_DualBlock = num_DualBlock
        self.linear_v = nn.Linear(v_dim, a_dim)
        self.relu = nn.ReLU()
        self.attention = AVGA()
        # self.attention = DotAttention()
        self.lstm_a_v = LSTM_A_V(a_dim=a_dim, v_dim=hidden_dim, hidden_dim=hidden_dim, category_num=category_num)
        self.psp = PSP(a_dim=a_dim*2, v_dim=hidden_dim*2)

        self.W1 = nn.Linear(2*hidden_dim, 1, bias=False)
        self.W2 = nn.Linear(64, 1, bias=False)
        
        self.W3 = nn.Linear(29, 1, bias=False)
        self.L1 = nn.Linear(2*hidden_dim, 64, bias=False)
        self.L2 = nn.Linear(64, category_num, bias=False)
        layers = [self.L1, self.L2]
        self.init_layers(layers)

    def init_layers(self, layers):
        for layer in layers:
            nn.init.xavier_uniform(layer.weight)

    def forward(self, audio, video, thr_val):
        # audio: [bs, 10, 128]
        # video: [bs, 10, 7, 7, 512]
        bs, seg_num, H, W, v_dim = video.shape
        fa_fea = audio
        video_t = self.attention(fa_fea, video) # [bs, 10, 512]
        video_t = self.linear_v(video_t) # [bs, 10, 128]
        lstm_audio, lstm_video = self.lstm_a_v(fa_fea, video_t)
        fusion, final_v_fea, final_a_fea, cross_v_fea, cross_a_fea, cross_att_v_to_a, cross_att_a_to_v = self.psp(lstm_audio, lstm_video, thr_val) # [bs, 10, 256]
        out = self.relu(self.L1(fusion))
        score = self.L2(out) #[bs, 10, 29]
        ######################################## weighting branch #######################
        temporal_wei = self.relu(self.W3(score)) # [bs, 10, 1]
        temporal_wei = torch.sigmoid(temporal_wei)
        score = score * temporal_wei.expand_as(score)
        #################################################################################
        out = score.permute(0, 2, 1) #[bs, 29, 10]
        out_avg = nn.AvgPool1d(out.size(2))(out).view(out.size(0), -1)
        out_avg = F.softmax(out_avg, dim=-1) # [bs, 29]

        return out_avg, score


if __name__ == "__main__":
    print("PSP model for weakly supervised AVE localization")
