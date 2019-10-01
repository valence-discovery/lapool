import torch
import numpy as np
import networkx as nx
import scipy.sparse as ss
import torch.nn.functional as F
from gnnpooling.utils import const
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdmolops
from rdkit.Chem.Draw import rdMolDraw2D
from torchvision.utils import make_grid
from torchvision import transforms
from gnnpooling.utils.tensor_utils import to_sparse
from PIL import Image
import io
import dgl


TRANSFORMER = transforms.ToTensor()
EDGE_DECODER = const.BOND_TYPES


def to_cuda(x):
    if const.CUDA_OK:
        return x.cuda()
    return x


def power_iteration(mat, n_iter=100):
    res = torch.ones(1, mat.shape[1]).expand(mat.shape[0], -1).unsqueeze(-1)    
    for i in range(n_iter):
        norm = res.norm(dim=1)
        res = torch.bmm(mat, res) 
        res = torch.div(res, norm.unsqueeze(-1))
    return res


def find_largest_eigval(mat, n_iter=100):
    if mat.dim() == 2:
        mat =  mat.unsqueeze(0)
    res = power_iteration(mat, n_iter=n_iter)
    # use the Rayleigh quotient.
    return torch.matmul(res.transpose(-1, -2), mat).matmul(res) / torch.matmul(res.transpose(-1, -2), res)


def restack(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, (tuple, list)):
        return torch.stack([x[i].squeeze() for i in range(len(x))], dim=0)
    raise ValueError("Cannot re-stack tensor")


def pad_graph(adj, max_num_node=None):
    """Returned a padded adj from an input adj"""
    num_node = adj.shape[0]
    old_shape = (adj.shape)
    if max_num_node in [-1, None]:
        return adj
    elif max_num_node < num_node:
        return adj[:max_num_node, :max_num_node]
    else:
        return F.pad(adj.transpose(-1, 0), (0, max_num_node - num_node) * 2, mode='constant', value=0).transpose(-1, 0)


def pad_feats(feats, no_atom_tensor=None, max_num_node=None):
    """Returned a padded adj from an input adj"""
    num_node = feats.shape[0]
    if max_num_node in [-1, None]:
        return feats
    elif max_num_node < num_node:
        return feats[:max_num_node, :max_num_node]
    else:
        if no_atom_tensor is None:
            no_atom_tensor = feats.new_zeros(feats.shape[-1])
        return torch.cat((feats, no_atom_tensor.unsqueeze(dim=0).expand(max_num_node - num_node, -1)), dim=0)


def get_adj_vector(adj):
    """Return a vector containing the upper triangle of an adj mat"""
    return adj[torch.ones_like(adj).triu().byte()]


def deg_feature_similarity(f1, f2):
    return 1 / (abs(f1 - f2) + 1)


def _add_index_to_mol(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber',
                                        str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def mol2im(mc, molSize=(300, 120), kekulize=False, outfile=None, im_type="png"):
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            pass
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    if im_type == "svg":
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])

    drawer.SetFontSize(drawer.FontSize() * 0.8)
    opts = drawer.drawOptions()
    for i in range(mc.GetNumAtoms()):
        opts.atomLabels[i] = mc.GetAtomWithIdx(i).GetSymbol() + str(i)
    opts.padding = 0.1
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()

    im = drawer.GetDrawingText()

    if im_type == "svg":
        im = im.replace('svg:', '')
        if outfile:
            with open(outfile, "w") as OUT:
                OUT.write(im)
    else:
        if outfile:
            drawer.WriteDrawingText(outfile)
    return im


def mol2svg(*args, **kwargs):
    return mol2im(*args, **kwargs, im_type="svg")


def mol2nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G


def add_index_to_mol(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def adj2nx(adj):
    return nx.from_numpy_matrix(adj.detach().cpu().numpy())


def decode_one_hot(encoding, fail=-1):
    if isinstance(encoding, np.integer):
        return encoding
    if sum(encoding) == 0:
        print("WTFFF")
        return fail
    return np.argmax(encoding)


def data2mol(data, atom_decoder):
    adj, x, *_ = data
    adj = restack(adj)
    x = restack(x)
    mols = []
    n_mols = adj.shape[0]
    for i in range(n_mols):
        mols.append(adj2mol(adj[i].cpu().detach().numpy(), x[i].cpu().detach().numpy(), atom_decoder))
    return mols

def get_largest_fragment(mol):
    return max(rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False), default=mol, key=lambda m: m.GetNumAtoms())


def adj2mol(adj_mat, atom_one_hot, atom_decoder, edge_decoder=EDGE_DECODER):
    mol = Chem.RWMol()
    atoms = [atom_decoder[decode_one_hot(atom)] for atom in atom_one_hot]
    n_atoms = adj_mat.shape[0]
    # contains edges, so get argmax on the edge
    edge_type = np.argmax(adj_mat, axis=-1)
    edges = np.triu(edge_type!=len(EDGE_DECODER), 1).nonzero() 
    accepted_atoms = {}
    for i, atom in enumerate(atoms):
        a = None
        if atom not in [None, "*"]:
            a = mol.AddAtom(Chem.Atom(atom))
        accepted_atoms[i] = a

    for start, end in zip(*edges):
        start = accepted_atoms.get(start)
        end = accepted_atoms.get(end)
        if (start is not None and end is not None):
            btype = edge_type[start, end]
            if btype < len(EDGE_DECODER):
                mol.AddBond(int(start), int(end), EDGE_DECODER[btype])
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        pass

    mol = mol.GetMol()
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception as e:
        print(e)
        mol = None
    if mol and mol.GetNumAtoms() > 0:
        return mol
    return None



def convert_mol_to_smiles(mollist, sanitize=False):
    smiles = []
    for mol in mollist:
        if sanitize:
            try:
                mol = Chem.SanitizeMol(mol)
            except:
                mol = None
        if mol:
            _ =[x.ClearProp('molAtomMapNumber') for x in mol.GetAtoms()]
            smiles.append(Chem.MolToSmiles(mol))
        else:
            smiles.append(None)
    return smiles


def sample_gumbel(x, eps=1e-8):
    noise = torch.rand_like(x)
    noise = -torch.log(-torch.log(noise + eps) + eps)
    return noise


def gumbel_sigmoid(logits, temperature):
    noise = sample_gumbel(logits)
    y = (logits + noise) / temperature
    y = F.sigmoid(y)  # , dim=-1)
    return y


def convert_to_grid(mols, molSize=(350, 200)):
    tensor_list = []
    for mol in mols:
        if mol:
            im = mol2im(mol, molSize=molSize)
            im = Image.open(io.BytesIO(im))
            tensor_list.append(TRANSFORMER(im.convert('RGB')))
    if tensor_list:
        return make_grid(torch.stack(tensor_list, dim=0), nrow=4)
    return None


def sample_sigmoid(logits, temperature=1, sample=False, thresh=0.5, gumbel=True, hard=False):
    y_thresh = torch.ones_like(logits) * thresh
    if gumbel:
        y = gumbel_sigmoid(logits, temperature)
    else:
        y = torch.sigmoid(logits)
        if sample:
            y_thresh = torch.rand_like(logits)
    if hard:
        return torch.gt(y, y_thresh).float()
    return y


def compute_deg_matrix(adj_mat, inv=False, selfloop=False):
    r"""
    Compute the inverse deg matrix from a tensor adjacency matrix

    Arguments
    ----------
        adj_mat: `torch.FloatTensor` of size (...)xNxN
            Input adjacency matrices (should not be normalized) that corresponds to a graph
        inv : bool, optional
            Whether the inverse of the computed degree matrix should be returned
            (Default value = False)
        selfloop: bool, optional
            Whether to add selfloop to the adjacency matrix or not

    Returns
    -------
        deg_mat: `torch.Tensor`
            Degree matrix  of the input graph    
    """
    if selfloop:
        adj_mat = torch.eye(adj_mat.shape[-1], device=adj_mat.device).expand_as(
            adj_mat) + adj_mat.clone()
    elif adj_mat.is_sparse:
        adj_mat = adj_mat.to_dense()
    deg_mat = torch.sum(adj_mat, dim=-2)
    if inv:
        # relying on the fact that it is a diag mat
        deg_mat = torch.pow(deg_mat, -0.5)
        deg_mat[torch.isinf(deg_mat)] = 0
    deg_mat = torch.diag_embed(deg_mat)
    return deg_mat, adj_mat


def inverse_diag_mat(mat, eps=1e-8):
    # add jitter to the matrix diag
    jitter = torch.eye(mat.shape[-1], device=mat.device).expand_as(mat)
    if torch.all(mat.masked_select(jitter.bool())>0):
        return mat.inverse() 
    return torch.inverse(jitter*eps + mat)
        

def to_symmetric(adj, max=True):
    r"""
    Convert an input matrix to a symmetric matrix

    The symmetric matrix is computed by doing :math:`A_{sym} (i,j) = max(A(i,j), A^T (i,j))`

    """
    if max:
        return torch.max(adj, adj.transpose(-2, -1))
    return (adj + adj.transpose(-2, -1))/2


def normalize_adj(adj, selfloop=True):
    r"""
    Symmetrically normalize a numpy adjacency matrix.
    See: https://github.com/tkipf/gcn/blob/master/gcn/utils.py#L122

    Arguments
    ----------
        adj: numpy.ndarray
            Adjacency matrix to be normalized
        selfloop: bool, optional
            Whether to add selfloop to the adjacency matrix or not

    Returns
    -------
        Normalized adjacency matrix
    """
    if isinstance(adj, torch.Tensor):
        deg_mat, adj_mat = compute_deg_matrix(adj, inv=True, selfloop=selfloop)
        norm_G = torch.matmul(
            deg_mat.detach(), adj_mat).matmul(deg_mat.detach())
        return norm_G

    def compute_norm(adj_i):

        if selfloop:
            adj_i = adj_i + ss.eye(adj_i.shape[0])
        adj_i = ss.coo_matrix(adj_i)
        rowsum = np.array(adj_i.sum(1))
        # this is possible since the deg matrix is a diag mat
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = ss.diags(d_inv_sqrt)
        norm_G = ((d_mat_inv_sqrt@adj_i)@d_mat_inv_sqrt).toarray()
        return norm_G

    multidim = len(adj.shape) == 3
    if multidim:
        return np.stack([compute_norm(x) for x in adj])
    return compute_norm(adj)


def get_path_length(adj_mat, k, strict=True):
    r"""
    Compute a set of path matrix for each adjacency matrix, that include all path  deg matrix from a tensor adjacency matrix

    .. warning::
        This method use matrix multiplication to find the k-path. It is for prototyping. 
        A less computationally expensive operation will be to do a breath-first search (linear)
        on each node and log the length of path visited

    Arguments
    ----------
        adj_mat: `torch.Tensor` of size (B,N,N)
            Input path matrix of length 1 (adjacency matrix).
        k : int
            Maximum path length to consider.
        strict: bool
            Whether to only keep the shortest path or to keep all.

    Returns
    -------
        path_matrix: `torch.Tensor` of size (k,B,N,N)
            Degree matrix  of the input graph    
    """
    prev_mat = adj_mat
    matlist = adj_mat.unsqueeze(0)
    diag_mat = torch.eye(
        prev_mat.shape[1], dtype=torch.uint8).expand_as(adj_mat)
    for i in range(2, k + 1):
        prev_mat = torch.bmm(prev_mat, adj_mat)
        if strict:
            no_path = (matlist.sum(dim=0) != 0).byte()
            new_mat = prev_mat.masked_fill(no_path, 0)
            new_mat.clamp_max_(1)
        else:
            new_mat = prev_mat
        # there os no point in receiving msg from the same node.
        new_mat[diag_mat] = 0
        matlist = torch.cat((matlist, new_mat.unsqueeze(0)), dim=0)
    return matlist


def pack_graph(batch_G, batch_x, return_sparse=False, fill_missing=0):
    r"""
    Pack a batch of graph and atom features into a single graph

    Arguments
    ----------
        batch_G: torch.LongTensor 2D iterable
            List of adjacency graph, each of size (n_i, n_i). 
            Sparse tensor allowed for faster computation too.
        batch_x: iterable of torch.Tensor 2D
            List of atom feature matrices, each of size (n_i, F), F being the number of features
        return_sparse (bool, optional): Whether to return a sparse graph at the end
            (Default value = False)
        fill_missing: int, optional
            fill out-of-graph bond with this value. 
            Note that filling out-of-graph positions, with anything besides 0, will disable sparsity.
            (Default value = 0)

    Returns
    -------
        new_batch_G, new_batch_x: torch.LongTensor 2D, torch.Tensor 2D
            This tuple represents a new arbitrary graph that contains the whole batch, 
            and the corresponding atom feature matrix. new_batch_G has a size (N, N), with :math:`N = \sum_i n_i`,
            while new_batch_x has size (N,D)
    """
    out_x = torch.cat(tuple(batch_x), dim=0)
    n_neigb = out_x.shape[0]
    # should be on the same device
    out_G = batch_G[0].new_zeros((n_neigb, n_neigb))
    cur_ind = 0
    for g in batch_G:
        g_size = g.shape[0] + cur_ind
        if g.is_sparse:  # the time advantage gained with sparse tensor might be lost here.
            g = g.to_dense()
        out_G[cur_ind:g_size, cur_ind:g_size] = g
        cur_ind = g_size

    if return_sparse and fill_missing == 0:
        out_G = to_sparse(out_G)
    return out_G, out_x  # .requires_grad_()


def batch_triu_to_full(adj, gsize, nedges=1):
    """Recover full matrix from upper triangular
    B, N*(N-1)/2, E == > B, N, N, E"""
    b_size = adj.shape[0]
    ind = np.ravel_multi_index(np.triu_indices(gsize, 1), (gsize, gsize))
    upper_tree = torch.zeros(gsize ** 2).index_fill(0, torch.from_numpy(ind), 1).contiguous().unsqueeze(-1).expand(gsize ** 2, nedges).byte()
    upper_tree = upper_tree.to(adj.device)
    adj_logits = adj.new_zeros(b_size, gsize ** 2, nedges)
    adj_logits = adj_logits.masked_scatter(upper_tree, adj).view(b_size, gsize, gsize, nedges)
    adj_logits = adj_logits + adj_logits.transpose(1, 2)
    return adj_logits



def batch_full_to_triu(adj, gsize):
    """Return upper triangular of a square matrix:
    B, N, N, E ==> B, N*(N-1)/2, E"""
    one_edges = torch.ones_like(adj)
    edges_mask = one_edges.permute(0, -1, 1, 2).triu(1) != 0 
    return adj.permute(0, -1, 1, 2).masked_select(edges_mask).contiguous().view(adj.shape[0], adj.shape[-1], -1).permute(0, -1, 1)


def adj_mat_from_edges(edges):
    adj = (edges.view(edges.shape[0], -1, edges.shape[-1]).argmax(dim=-1) != (edges.shape[-1] - 1)).long()
    return adj.view(edges.shape[0], edges.shape[1], edges.shape[2]).float()


def dgl_from_edge_matrix(G, X, mask=None, full_mat=False):
    g_collections = []
    adj_mat = adj_mat_from_edges(G)
    for i, (g_i, e_i, x_i) in enumerate(zip(adj_mat, G, X)):
        g = dgl.DGLGraph()
        dt = {'h':x_i}
        if mask is not None:
            dt.update(mask=mask[i])
        g.add_nodes(g_i.shape[-1], dt)
        adj_i = torch.ones_like(g_i)
        edge_i = e_i.view(-1, e_i.shape[-1])
        if not full_mat:
            adj_i = g_i
            edge_i = e_i.masked_select((adj_i > 0).unsqueeze(-1)).view(-1, e_i.shape[-1])                
        g.add_edges(*zip(*adj_i.nonzero()))
        g.edata['he'] = edge_i
        g_collections.append(g)
    return g_collections
    
