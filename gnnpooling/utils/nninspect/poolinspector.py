import torch
import numpy as np
import os
import io
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import networkx as nx
import subprocess
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from collections import defaultdict as ddict
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _get_n_colors(n, seed=None):
    """Get a n colors from the color palette"""
    RGB_COLORS = sns.color_palette(sns.hls_palette(n, h=0.02, l=0.55, s=0.55)) 
    np.random.seed(seed)
    np.random.shuffle(RGB_COLORS)
    return RGB_COLORS[:n]

def _adj_to_nx(adj):
    """Convert an adjacency matrix to networkx"""
    return nx.from_numpy_matrix(adj)


def _get_text_colors(colors):
    """Get Text color that should match with a given background"""
    tcols = [(r * 299 + g * 587 + b * 114)*1.0 /
             1000 > 0 for (r, g, b) in colors]
    return ["#DDDDDD" if x < 123 else "#222222" for x in tcols]


def _position_communities(g, partition, n_edges, spring_layout=True, **kwargs):
    """Compute optimal positions for the two graphs"""
    ci, cj = set(partition.values())  # 2
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from([ci, cj])
    hypergraph.add_edge(ci, cj, weight=n_edges)
    # find layout for communities
    if spring_layout:
        pos_communities = nx.spring_layout(hypergraph, **kwargs)
    else:
        pos_communities = nx.bipartite_layout(hypergraph, [ci], **kwargs)
    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]
    return pos


def _position_nodes(g, partition, **kwargs):
    """Compute optimal node positions, given their community"""
    communities = ddict(list)
    for node, community in partition.items():
        communities[community].append(node)
    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        # neato as alternative
        #pos_subgraph = nx.nx_agraph.graphviz_layout.graphviz_layout(subgraph, prog='dot')
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)
    return pos


def community_layout(g, partition, inter_edges):
    """Comunity layout for plotting node assignment (projection) into clusters"""
    pos_communities = _position_communities(
        g, partition, inter_edges, scale=3.)
    pos_nodes = _position_nodes(g, partition, scale=1.)
    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]
    return pos


def _get_ax_size(ax, fig, margin=0):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return int(width)-margin, int(height)-margin


def _link_graphs(G1, G2, mapper, leaders, cpalette=[]):
    G1, G2 = _adj_to_nx(G1), _adj_to_nx(G2)
    new_G = nx.disjoint_union(G1, G2)
    partition = dict((i, (i-G1.number_of_nodes()+1 > 0))
                     for i in range(new_G.number_of_nodes()))

    def new_index(x): return x + G1.number_of_nodes()
    color_palette = np.asarray([matplotlib.colors.to_hex(rgb) for rgb in cpalette]) # f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{(rgb[2]*255):02x}", _get_n_colors(mapper.shape[-1]
    node_colors = np.full(sum(mapper.shape), "#222222")
    if leaders is not None and len(leaders) > 0:
        node_colors[leaders] = color_palette
    node_colors[-len(color_palette):] = color_palette
    inter_edges = []
    for n_i in range(mapper.shape[0]):
        for n_j in range(mapper.shape[1]):
            new_G.add_edge(n_i, new_index(
                n_j), weight=mapper[n_i, n_j], etype="inter")
            inter_edges.append((n_i, new_index(n_j)))
    return new_G, partition, node_colors, inter_edges


def _to_np(tensor):
    if tensor is None:
        return tensor
    return tensor.squeeze(0).detach().cpu().numpy()


class PoolingInspector:

    def __init__(self, arch, show=False, outdir="pool_inspect", freq=1, thresh=0, figsize=(15, 12), speed=0.01, save_video=True, draw_labels=True):
        self.fig = plt.figure(constrained_layout=False, figsize=figsize)
        gs = self.fig.add_gridspec(ncols=2, nrows=2, figure=self.fig, width_ratios=[
                                   5, 4], height_ratios=[5, 2])
        self.mol_ax = self.fig.add_subplot(gs[1, 0])
        self.mol_ax.set_title('Molecule')
        self.pie_ax = self.fig.add_axes(self.mol_ax.get_position())
        self.mapper_ax = self.fig.add_subplot(gs[1, 1])
        self.mapper_ax.set_title('Assignment')
        self.mapper_ax.xaxis.tick_top()
        self.mapper_ax.tick_params(labelbottom=False,labeltop=True)

        divider = make_axes_locatable(self.mapper_ax)
        self.cax = divider.append_axes('bottom', size='5%', pad="4%") 

        self.graph_ax = self.fig.add_subplot(gs[0, :])
        self.graph_ax.set_title('Projection')
        self.fig.tight_layout()
        self.draw_labels = draw_labels
        self.iter = 0
        self.freq = freq
        self.std_thresh = thresh
        self.outdir = outdir
        self.arch = arch
        self.show = show
        self.saving_n = 0
        self.speed = speed
        self.video = save_video
        if not self.show:
            plt.ioff()
            # matplotlib.use('Agg')

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        plt.show(block=False)


    def __call__(self, pooler, G, new_G, mol=None, training=False):
        if self.iter % self.freq == 0:
            leaders = []
            if hasattr(pooler, "leader_idx"):
                leaders = pooler.leader_idx.detach().cpu().numpy()
            self.draw(mol, _to_np(G), _to_np(new_G), _to_np(
                pooler.cur_S), leaders, training=training)
        self.iter += 1

    def __del__(self):
        """Force saving to video before closing"""
        if self.video:
            try:
                self.to_video(outfile=self.arch)
            except:
                pass
        plt.close(self.fig)

    def draw(self, mol, G1, G2, mapper, leaders, training=False):
        self._clear_axis()
        seed = np.random.randint(100)
        color_palette = _get_n_colors(mapper.shape[-1], seed)
        if len(mapper.shape) == 1:
            mapper = mapper[:, np.newaxis]
        self._draw_mapper(mapper, cpalette=color_palette)
        self._draw_mol(mol, mapper, leaders, std_thresh=self.std_thresh, cpalette=color_palette)
        self._draw_graphs(G1, G2, mapper, leaders, cpalette=color_palette)
        self.fig.suptitle("Training" if training else "Testing",
                          fontsize=14, color={1: "blue", 0: "red"}[training])
        
        self.save()
        if self.show:
            plt.pause(self.speed)

    def _clear_axis(self):
        """Clear the axis of the figure"""
        self.mol_ax.clear()
        self.pie_ax.clear()
        self.mapper_ax.clear()
        self.graph_ax.clear()
        self.pie_ax.set_xticks([], [])
        self.pie_ax.set_yticks([], [])
        self.mol_ax.set_xticks([], [])
        self.mol_ax.set_yticks([], [])
        self.mol_ax.axis('off')
        self.graph_ax.set_xticks([], [])
        self.graph_ax.set_yticks([], [])

    def save(self):
        self.fig.savefig(os.path.join(
            self.outdir, str(self.saving_n)+".svg"), dpi=150)
        self.saving_n += 1

    def folder_clean(self):
        files = glob.glob(os.path.join(self.outdir, "*.png"))
        for f in files:
            os.unlink(f)

    def to_video(self, outfile=None, interval=1, clean=False):
        outfile = outfile or self.arch
        outfile = os.path.join(self.outdir, "{}.mp4".format(outfile))
        try:
            if os.path.exists(outfile):
                os.unlink(outfile)
            subprocess.call([
                'ffmpeg', '-i', os.path.join(self.outdir, '%d.png'), '-r', str(
                    interval), '-pix_fmt', 'yuv420p', outfile
            ])
        except Exception as e:
            print(e)
            print("Cannot save video ...")
        if clean:
            self.folder_clean()

    def _draw_mapper(self, mapper, clim=(0, 1), cpalette=[]):
        if clim is None:
            maxval = np.max(np.abs(mapper))
            clim = (-maxval, maxval)
        mapper = mapper.T
        m, n = mapper.shape
        im = self.mapper_ax.imshow(mapper, clim=clim, cmap=plt.cm.gray)
        self.mapper_ax.set_yticklabels('')
        self.mapper_ax.set_yticks([i for i in range(m)], minor=False)
        self.mapper_ax.set_yticklabels([f'C-{i}' for i in range(m)], minor=False)
        ticks = list(self.mapper_ax.get_yticklabels())
        for i in range(m):
            ticks[i].set_color(cpalette[i])
        #self.cax.tick_params(labelbottom='off',labeltop='on')

        self.fig.colorbar(im, cax=self.cax, orientation='horizontal')

    def _draw_graphs(self, G1, G2, mapper, leaders, cpalette=[], **kwargs):
        ax = self.graph_ax
        G, partition, cols, inter_edges = _link_graphs(G1, G2, mapper, leaders, cpalette=cpalette)
        pos = community_layout(G, partition, len(inter_edges),  **kwargs)

        cmap = plt.cm.get_cmap('gist_yarg')
        colors = [cmap(min(1.0, max(0.0, G[u][v]["weight"])) )
                  for u, v in inter_edges]
        # normalize item number values to colormap
        arcs = nx.draw_networkx_edges(
            G, pos=pos, edgelist=inter_edges, edge_color=colors, style="dashed", ax=ax, width=1.5)
        #nx.draw_networkx_edge_labels(G, pos, label_pos=0.5, font_size=5, font_color='k', ax=ax, rotate=True)
        # G1 nodes
        G1_nodes = [x for x, v in partition.items() if v == 0]
        G2_nodes = [x for x, v in partition.items() if v == 1]

        nodes_G1 = nx.draw_networkx_nodes(G, pos, G1_nodes, node_size=70, node_color=[
                                          cols[i] for i in G1_nodes], ax=ax)
        nx.draw_networkx_labels(G, pos, labels=dict(
            zip(leaders, leaders)), font_size=7, font_color='#111111', ax=ax)
        # G2 nodes
        nodes_G2 = nx.draw_networkx_nodes(G, pos, G2_nodes, node_size=70, node_shape='s', linewidths=2, node_color=[
                                          cols[i] for i in G2_nodes], ax=ax)
        # draw regular edges
        reg_edges = [e for e in G.edges if not (
            e[::-1] in inter_edges or e in inter_edges)]
        w = [G[e[0]][e[1]]["weight"] for e in reg_edges]
        bonds = nx.draw_networkx_edges(G, pos=pos, edgelist=reg_edges,
                                       edge_color="#222222", style="solid", arrows=False, width=w, ax=ax)
        # draw inter connecting edges
        if self.draw_labels:
            e_labels = dict([((e[0],e[1]), "{:.2f}".format(G[e[0]][e[1]]["weight"])) for e in G.edges if e[0] in G2_nodes and e[1] in G2_nodes])
            nx.draw_networkx_edge_labels(G, pos, edge_labels=e_labels, font_size=7, sfont_color="black", ax=ax)


    def _draw_mol(self, mol, mapper, leaders, std_thresh=0, cpalette=[]):
        size = _get_ax_size(self.mol_ax, self.fig)
        tot_cluster = mapper.shape[-1]
        single_contribution = (mapper - mapper.max(axis=-1, keepdims=True)[
                               0] + std_thresh*np.std(mapper, axis=-1, keepdims=True) > 0).sum(axis=-1)

        # Prepare the molecule drawer
        rdDepictor.Compute2DCoords(mol)
        mol = rdMolDraw2D.PrepareMolForDrawing(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.SetFontSize(drawer.FontSize()*0.85)

        # Set the drawing options and get the atom labels
        opts = drawer.drawOptions()
        opts.clearBackground = True

        opts.padding = 0.01
        atomLabels = []
        for i in range(mol.GetNumAtoms()):
            atomLabels.append(mol.GetAtomWithIdx(i).GetSymbol()+"$_{"+str(i)+"}$")
        opts.setAtomPalette({-1: (0, 0, 0)})
         
        # Draw the molecule to a buffer, then read the buffer and display it in the mol_ax
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        png = mpimg.imread(io.BytesIO(png), format='PNG')
        self.mol_ax.axis('off')
        self.mol_ax.imshow(png, origin="upper", aspect="equal",
                           interpolation="lanczos", zorder=-1)  # lanczos")

        # Get all the atom coordinates, and flip the x axis 
        coors = []
        for atom in mol.GetAtoms():
            a_idx = atom.GetIdx()
            coor = np.array(list(drawer.GetDrawCoords(a_idx)))
            coor[1] = png.shape[1] - coor[1]
            coors.append(coor)

        # Get the median minimum distance between atoms to decide the font size and pie-chart radius
        dist_all = np.zeros((len(coors), len(coors)))
        for ii in range(len(coors)):
            for jj in range(ii, len(coors)):
                dist_all[ii][jj] = (np.sqrt(np.sum((coors[ii] - coors[jj])**2)))
        dist_all += np.eye(len(coors)) * 1000
        dist_mins = np.min(dist_all + np.transpose(dist_all), axis=0)
        min_dist = np.median(dist_mins)
        min_dist = max(min_dist, 25)

        # Find the starting and ending coordinates of the axis
        coors = np.array(coors)
        mins = np.min(coors, axis=0)
        maxs = np.max(coors, axis=0)
        coors_shape = maxs - mins
        coors_pad = opts.padding * coors_shape
        coors_start = -coors_pad - 0.5
        coors_end = coors_shape + coors_pad - 0.5
        
        # Display the pie charts, along with the atom labels
        self.fig.sca(self.pie_ax)
        radius = min_dist/3
        fontsize = radius * 0.8
        fontsize = max(min(fontsize, 20), 14)
        for a_idx in range(coors.shape[0]):
            if a_idx < mapper.shape[0]:
                center = coors[a_idx] - mins
                if a_idx in leaders:
                    self.pie_ax.pie([1], colors={'black'}, radius=radius*1.25, center=center)
                self.pie_ax.pie(mapper[a_idx], colors=cpalette, radius=radius, center=center)
                self.pie_ax.text(center[0], center[1], atomLabels[a_idx], fontsize=fontsize, 
                    horizontalalignment='center', verticalalignment='center')
        
        # Reset the pie_ax position and boundaries
        self.pie_ax.set_position(self.mol_ax.get_position())
        self.pie_ax.set_xlim(coors_start[0], coors_end[0])
        self.pie_ax.set_ylim(coors_start[1], coors_end[1])
