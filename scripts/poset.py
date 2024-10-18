"""
This script implements the filtrations by posets introduced in the "Sequents, barcodes, and homology" manuscript.
Author: Negin Karisani
Date: May 2022
"""

import networkx as nx
import igviz as ig
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# from CRISPR.lib import Lib, Var
from scripts.my_library import MyLib

class Poset:

    def __init__(self, gamma_df, delta_df, poset=None, filename=''):
        self.gamma = list(gamma_df.columns)
        self.delta = list(delta_df.columns)
        self.gamma_data_df = gamma_df
        self.delta_data_df = delta_df
        self.samples_count = gamma_df.shape[0]
        if poset is None:
            self.poset_nx = self.create_poset(filename)
        else:
            self.poset_nx = poset

    @staticmethod
    def create_barcode_sequents(gamma_df, delta_df, poset_nx=None, KEEP_TRIVIAL=False, filename=''):
        sp = Poset(gamma_df, delta_df, poset_nx, filename)
        return sp.create_barcode_for_minimal_nodes(KEEP_TRIVIAL)

    def create_poset(self, filename, visualize_poset=False):
        """
           The poset is created using a bottom-up approach.
        """
        self.poset_nx = nx.DiGraph() # A poset is a directed graph with edges pointing to top-level nodes.
        total_no = len(self.gamma) + len(self.delta)
        cur_id = 0
        cur_node = (cur_id, {'gamma': frozenset([]), 'delta': frozenset([]), 't': 1})
        self.poset_nx.add_nodes_from([cur_node])
        ids_dic = {frozenset([]): cur_id}
        id_counter = 1
        node_queue =[]
        while len(cur_node[1]['delta'].union(cur_node[1]['gamma'])) != total_no: # Stop when reaching the top node
            print(len(cur_node[1]['delta'].union(cur_node[1]['gamma'])), ' / ', total_no)

            # The top neighbors of the current node vary by augmenting the delta and gamma elements.
            aug_gamma = [cur_node[1]['gamma'].union(frozenset([ele])) for ele in self.gamma if ele not in cur_node[1]['gamma']]
            aug_delta = [cur_node[1]['delta'].union(frozenset([ele])) for ele in self.delta if ele not in cur_node[1]['delta']]
            nodes = []
            edges = []

            # Create top neighbors that vary based on gamma elements and calculate their timepoints.
            for new_gamma in aug_gamma:
                all_element = new_gamma.union(cur_node[1]['delta'])
                if all_element not in ids_dic:
                    ids_dic[all_element] = id_counter
                    cur_t = self.get_t_filtration2(new_gamma, cur_node[1]['delta'])
                    # cur_t = self.get_t_filtration2_multiset(new_gamma, cur_node[1]['delta'])
                    new_node = (id_counter, {'gamma': new_gamma, 'delta': cur_node[1]['delta'], 't': cur_t})
                    nodes.append(new_node)
                    edges.append((cur_id, id_counter))
                    node_queue.append(new_node)
                    id_counter += 1
                else:
                    temp_id = ids_dic[all_element]
                    edges.append((cur_id, temp_id))

            # Create top neighbors that vary based on delta elements and calculate their timepoints.
            for new_delta in aug_delta:
                all_element = new_delta.union(cur_node[1]['gamma'])
                if all_element not in ids_dic:
                    ids_dic[all_element] = id_counter
                    cur_t = self.get_t_filtration2(cur_node[1]['gamma'], new_delta)
                    # cur_t = self.get_t_filtration2_multiset(cur_node[1]['gamma'], new_delta)
                    new_node = (id_counter, {'gamma': cur_node[1]['gamma'], 'delta': new_delta, 't': cur_t})
                    nodes.append(new_node)
                    edges.append((cur_id, id_counter))
                    node_queue.append(new_node)
                    id_counter += 1
                else:
                    temp_id = ids_dic[all_element]
                    edges.append((cur_id, temp_id))

            self.poset_nx.add_nodes_from(nodes)
            self.poset_nx.add_edges_from(edges)

            cur_node = node_queue.pop(0)
            cur_id = cur_node[0]
        if visualize_poset:
            self.visualize_poset()
        print(f'number of elements: {self.poset_nx.number_of_nodes()}')
        print(f'number of relations: {self.poset_nx.number_of_edges()}')
        MyLib.save_pkl(self.poset_nx, f'poset_{filename}.pkl')
        return self.poset_nx

    def get_t_filtration2(self, cur_gamma, cur_delta):
        """
            The input datasets should be binary
        """
        left_df = self.gamma_data_df[cur_gamma]
        X = left_df[left_df == 1]  # Change zeros to NaNs
        X = X[X.isnull().any(axis=1)]  # Select any row with NaN value
        if X.shape[1] != 0:
            gamma_samples = set(X.index.tolist())
        else:
            gamma_samples = set([])

        right_df = self.delta_data_df[cur_delta]
        Y = right_df[right_df == 0]
        Y = Y[Y.isnull().any(axis=1)]
        if Y.shape[1] != 0:
            delta_samples = set(Y.index.tolist())
        else:
            delta_samples = set([])

        samples = gamma_samples.union(delta_samples)
        t = 1 - len(samples) * 1/self.samples_count
        # t = 1 - t ** 2
        # t = t ** 2
        # t = (t-1) ** 2
        # t = 1/(t+0.001)
        return t

    def get_t_filtration2_multiset(self, cur_gamma, cur_delta):
        """
            In case the input dataframes have more than one category
        """
        t_l = []
        if self.gamma_data_df[cur_gamma].shape[1] != 0:
            lim = int(max(self.gamma_data_df[cur_gamma].max())) + 1
        else:
            lim = 2
        for i in range(1, lim):
            left_df = self.gamma_data_df[cur_gamma]
            X = left_df[left_df == i]  # Change any value other than i to NaN
            X = X[X.isnull().any(axis=1)]  # Select any row with NaN value
            if X.shape[1] != 0:
                gamma_samples = set(X.index.tolist())
            else:
                gamma_samples = set([])

            right_df = self.delta_data_df[cur_delta]
            Y = right_df[right_df == i]
            Y = Y[Y.any(axis=1)]
            if Y.shape[1] != 0:
                delta_samples = set(Y.index.tolist())
            else:
                delta_samples = set([])

            samples = gamma_samples.union(delta_samples)
            t = 1 - len(samples) * 1/self.samples_count
            t_l.append(t)
        t = max(t_l)
        return t

    def create_barcode_for_minimal_nodes(self, KEEP_TRIVIAL=False):
        """
            Identifies minimal nodes in the poset and creates their bars
        """
        minimals_dic = dict() # A dictionary whose keys are the start timepoints of bars, and whose values are tuples of the form (node_id, end_timepoint).
        node_queue = [0]
        while len(node_queue) > 0:
            cur_id = node_queue.pop(0)
            cur_t = self.poset_nx.nodes(data=True)[cur_id]['t']
            is_minimal = True
            min_t = np.infty
            for u, v, data in self.poset_nx.in_edges(cur_id, data=True):
                # A node is not minimal if it has at least one child whose timepoint is less than or equal.
                if self.poset_nx.nodes(data=True)[u]['t'] <= cur_t:
                    is_minimal = False
                # Keep track of the minimum timepoint of the children, which corresponds to the bar's end timepoint.
                if self.poset_nx.nodes(data=True)[u]['t'] < min_t:
                    min_t = self.poset_nx.nodes(data=True)[u]['t']
            if is_minimal:
                if cur_t in minimals_dic:
                    minimals_dic[cur_t].append((cur_id, min_t))
                else:
                    minimals_dic[cur_t] = [(cur_id, min_t)]
            node_queue += [v for u, v in self.poset_nx.out_edges(cur_id) if v not in node_queue]

        bars = []
        count = 1
        max_end_t = 0
        bars_length_dic = dict()

        for t in sorted(minimals_dic):
            for node, end_t in minimals_dic[t]:
                if KEEP_TRIVIAL: # Keep bars where one side of their sequence is an empty set.
                    count, max_end_t, bars, bars_length_dic = self.create_bars(count, t, node, end_t, max_end_t, bars,
                                                                         bars_length_dic)
                elif  len(self.poset_nx.nodes(data=True)[node]['gamma']) != 0 and\
                        len(sorted(self.poset_nx.nodes(data=True)[node]['delta'])) != 0:
                    count, max_end_t, bars, bars_length_dic = self.create_bars(count, t, node, end_t, max_end_t, bars,
                                                                        bars_length_dic)

        return (bars, bars_length_dic, max_end_t)

    def create_bars(self, count, t, node, end_t, max_end_t, bars, bars_length_dic):
        if end_t > max_end_t and end_t != np.infty:
            max_end_t = end_t
        text = f"{sorted(self.poset_nx.nodes(data=True)[node]['gamma'])} |-- {sorted(self.poset_nx.nodes(data=True)[node]['delta'])}"
        cur_length = end_t - t
        bars.append((count, [t, end_t], text))
        if cur_length in bars_length_dic:
            bars_length_dic[cur_length].append((text, t, end_t))
        else:
            bars_length_dic[cur_length] = [(text, t, end_t)]
        count += 1
        return (count, max_end_t, bars, bars_length_dic)

    @staticmethod
    def summarize_barcode_and_visualize(bars, bars_length_dic, max_end_t, top_bars_cutoff=6, top_bars_cutoff_print=15,
                                        fontsize_text=7, fontsize_ticks=12, fontsize_axis=16,
                                        figure_width=10, figure_height=7, filename=''):

        lengths = sorted(bars_length_dic.keys(), reverse=True)

        # print top bars m bars
        m = min(top_bars_cutoff_print, len(lengths))
        for count, k in enumerate(lengths):
            print(f'\n{count}=======bar length: {round(k, 3)} ======')
            for text, t, end_t in bars_length_dic[k]:
                print(f'[{round(t, 3)}, {round(end_t, 3)}]')
                print(text)
            if count == m:
                break
        if len(lengths) > 0:
            # annotate top n bars in the figure
            n = min(top_bars_cutoff, len(lengths) - 1)
            thershold = lengths[n]
            Poset.visualize_barcode(bars, max_end_t, thershold, fontsize_text=fontsize_text, figure_width=figure_width,
                                    figure_height=figure_height, fontsize_ticks=fontsize_ticks, fontsize_axis=fontsize_axis,
                                    filename=filename)
        else:
            print('No nontrivial bar is found!')

    @staticmethod
    def visualize_barcode(bars, x_lim, thershold=None, fontsize_text=7, fontsize_ticks=10, fontsize_axis=12,
                          figure_width=10, figure_height=7, filename=''):

        plt.figure(figsize=(figure_width, figure_height))
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.xlim(0, x_lim+0.01)#
        plt.ylim(0, len(bars))
        plt.xlabel('t (Filtration Type II)', fontsize=fontsize_axis)
        plt.ylabel('Bars', fontsize=fontsize_axis)
        plt.grid(False)
        # plt.grid(which='both')
        texts = []
        for y, xs, t in bars:
            c = 'green'
            if np.inf not in xs:
                diff = xs[1] - xs[0]
                if diff < thershold:
                    t = ''
                plt.annotate('', xy=(xs[0], y), xycoords='data', xytext=(xs[1], y),
                             arrowprops=dict(arrowstyle='-', color=c, lw=2, shrinkA=0, shrinkB=0))
                plt.annotate(t, xy=(xs[1], y), xycoords='data', xytext=(-5, 5), textcoords='offset points',
                             fontsize=fontsize_text, va='baseline', ha='right', color=c)
        #     texts.append(plt.annotate(t, xy=(xs[1], y), xycoords='data', xytext=(-5, 5), textcoords='offset points',
        #                  fontsize=12, va='baseline', ha='right', color=c))
        # adjust_text(texts, only_move={'points': 'xy', 'texts': 'xy'}, force_points=0.001, force_text=0.2,
        #             # autoalign=True,
        #             arrowprops=dict(arrowstyle="->", color='purple', alpha=0.25, lw=0.3))
        # plt.show()
        plt.savefig(f"output_files/barcode_{filename}.jpeg", dpi=600)

    def visualize_poset(self):
        fig = ig.plot(
            self.poset_nx,  # Your graph
            # title=,
            size_method="static",  # sizing_list,  # , # Makes node sizes the same
            color_method= "#ffcccb",  # "#ffcccb", # Makes all the node colours black,
            node_text=["gamma", "delta", "t"],  # Adds the 'prop' property to the hover text of the node
            # annotation_text="Visualization made by <a href='https://github.com/Ashton-Sidhu/plotly-graph'>igviz</a> & plotly.", # Adds a text annotation to the graph
            # node_label="prop",  # Display the "prop" attribute as a label on the node
            # node_label_position="top center",  # Display the node label directly above the node
            # edge_text=['jaccard_distance'],  # Display the "edge_prop" attribute on hover over the edge
            # edge_label="similarity_ratio",  # Display the "edge_prop" attribute on the edge
            # edge_label_position="bottom center",  # Display the edge label below the edge
            layout="circular"
            # arrow_size=2
        )
        fig.update_layout(
            title_text="Sequent Poset "
        )
        fig.write_html('output_files/sequent_poset' + '.html', auto_open=True)

class DepMapExperiments:

    @staticmethod
    def process_data(feat_df, target_df, onco_l, right_l):

        exts = [
            # '_CN',
            # '_Exp',
            '_Hot',
            '_Dam',
            '_NonCon',
        ]
        features = []
        for ext in exts:
            features += [s+ext for s in onco_l if s+ext in feat_df.columns]
        # features = []
        # for ext in exts:
        #     features += [s for s in feat_df.columns if s.endswith(ext)]
        gamma_df = feat_df[features].copy()
        if '_Exp' in exts:
            gamma_df = DepMapExperiments.calc_gene_expression_lbls(gamma_df)
        features = [s for s in right_l if s in target_df.columns]
        delta_df = target_df[features].copy()
        print(f'gamma shape: {gamma_df.shape}')
        print(f'delta shape: {delta_df.shape}')
        return gamma_df, delta_df

    @staticmethod
    def calc_gene_expression_lbls(feat_df):
        exps = [col for col in feat_df.columns if col.endswith('_Exp')]
        feat_df[exps] = feat_df[exps].apply(stats.zscore)
        feat_df[exps] = feat_df[exps].apply(DepMapExperiments.construct_lbls)
        return feat_df

    @staticmethod
    def construct_lbls(exp_value):
        low = 1
        zeros = exp_value[exp_value <= low]
        ones = exp_value[exp_value > low]
        exp_value[zeros.index] = 0
        exp_value[ones.index] = 1
        return exp_value

    # @staticmethod
    # def process_data_target(feat_df, target_df, left_l, right_l):
    #     exts = [
    #         '_Hot',
    #         '_Dam',
    #         '_NonCon',
    #     ]
    #     features = []
    #     for ext in exts:
    #         features += [s + ext for s in left_l if s + ext in feat_df.columns]
    #     target_feats = []
    #     for target in right_l:
    #         print('\n', target)
    #         features_targets = Var.get_features('all,targets', target, 100)
    #
    #         features_targets = features_targets[:3]
    #         print(features_targets)
    #         target_feats = DepMapExperiments.merge_lists(features_targets, target_feats)
    #
    #     gamma_df = pd.concat([feat_df[features], target_df[target_feats]], axis=1)
    #
    #     features = [s for s in right_l if s in target_df.columns]
    #     delta_df = target_df[features].copy()
    #
    #     print(f'gamma shape: {gamma_df.shape}')
    #     print(f'delta shape: {delta_df.shape}')
    #     return gamma_df, delta_df
    #
    # @staticmethod
    # def merge_lists(list1, list2):
    #     new_set = set(list1).union(set(list2))
    #     return sorted(new_set)
    #
    # @staticmethod
    # def get_onco_info(feat_f, target_f, onco_f):
    #     feat_df = MyLib.load_h5py(feat_f)
    #     target_df = MyLib.load_h5py(target_f)
    #     onco_l = MyLib.load_csv(onco_f)['gene'].tolist()
    #     left_l = ['HRAS (3265)', 'KRAS (3845)', 'BRAF (673)',
    #               'NRAS (4893)']  # ['BRAF (673)','TP53 (7157)',  'EDA2R (60401)','RHEBL1 (121268)', 'MIA (8190)']
    #     right_l = ['HRAS (3265)', 'KRAS (3845)', 'BRAF (673)', 'NRAS (4893)']  # , 'SHOC2 (8036)'
    #     gamma_df, delta_df = DepMapExperiments.process_data(feat_df, target_df, left_l, right_l)
    #
    #     print('------------------------------------')
    #     print('Delta features')
    #     if onco_l is not None:
    #         feat_l = onco_l
    #     else:
    #         feat_l = delta_df.columns.tolist()
    #     for col in feat_l:
    #         print(col, delta_df[col].sum())
    #
    #     print('------------------------------------')
    #     print('Gamma features')
    #     sum_sr = gamma_df.sum()
    #     sum_sr = sum_sr.sort_values(ascending=False)
    #     print(sum_sr)





