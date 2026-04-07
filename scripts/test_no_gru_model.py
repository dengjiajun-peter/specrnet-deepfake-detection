import sys
sys.path.insert(0, '.')
from model import SpecRNet

conf = {
    'filts': [1, [1, 20], [20, 64]],
    'gru_node': 64,
    'nb_gru_layer': 2,
    'nb_fc_node': 64,
    'nb_classes': 2,
}

m = SpecRNet(conf, variant='no-gru', device='cpu')
print('Constructed no-gru model; param count:', sum(p.numel() for p in m.parameters()))

m2 = SpecRNet(conf, variant='default', device='cpu')
print('Constructed default(model with GRU); param count:', sum(p.numel() for p in m2.parameters()))

m3 = SpecRNet(conf, variant='gap', device='cpu')
print('Constructed gap model; param count:', sum(p.numel() for p in m3.parameters()))
