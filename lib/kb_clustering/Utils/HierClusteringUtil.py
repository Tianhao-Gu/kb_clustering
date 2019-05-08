
import time
import json
import os
import logging
import errno
import uuid
import sys
import shutil
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import itertools
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import plot
import scipy.cluster.hierarchy as hier


from installed_clients.DataFileUtilClient import DataFileUtil
from installed_clients.GenericsAPIClient import GenericsAPI
from installed_clients.kb_ke_utilClient import kb_ke_util


class HierClusteringUtil:

    METRIC = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine",
              "dice", "euclidean", "hamming", "jaccard", "kulsinski", "matching",
              "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath", "sqeuclidean",
              "yule"]

    METHOD = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]

    CRITERION = ["inconsistent", "distance", "maxclust"]

    def _mkdir_p(self, path):
        """
        _mkdir_p: make directory for given path
        """
        if not path:
            return
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    def _validate_run_hierarchical_cluster_params(self, params):
        """
        _validate_run_hierarchical_cluster_params:
                validates params passed to run_hierarchical_cluster method
        """

        logging.info('start validating run_hierarchical_cluster params')

        # check for required parameters
        for p in ['matrix_ref', 'workspace_name', 'cluster_set_name']:
            if p not in params:
                raise ValueError('"{}" parameter is required, but missing'.format(p))

        # check metric validation
        metric = params.get('dist_metric')
        if metric and metric not in self.METRIC:
            error_msg = 'INPUT ERROR:\nInput metric function [{}] is not valid.\n'.format(metric)
            error_msg += 'Available metric: {}'.format(self.METRIC)
            raise ValueError(error_msg)

        # check method validation
        method = params.get('linkage_method')
        if method and method not in self.METHOD:
            error_msg = 'INPUT ERROR:\nInput linkage algorithm [{}] is not valid.\n'.format(
                                                                                        method)
            error_msg += 'Available metric: {}'.format(self.METHOD)
            raise ValueError(error_msg)

        # check criterion validation
        criterion = params.get('fcluster_criterion')
        if criterion and criterion not in self.CRITERION:
            error_msg = 'INPUT ERROR:\nInput criterion [{}] is not valid.\n'.format(criterion)
            error_msg += 'Available metric: {}'.format(self.CRITERION)
            raise ValueError(error_msg)

    def __init__(self, config):
        self.ws_url = config["workspace-url"]
        self.callback_url = config['SDK_CALLBACK_URL']
        self.token = config['KB_AUTH_TOKEN']
        self.shock_url = config['shock-url']
        self.srv_wiz_url = config['srv-wiz-url']
        self.scratch = config['scratch']

        # helper kbase module
        self.dfu = DataFileUtil(self.callback_url)
        self.ke_util = kb_ke_util(self.callback_url, service_ver="dev")
        self.gen_api = GenericsAPI(self.callback_url, service_ver="dev")

        plt.switch_backend('agg')
        sys.setrecursionlimit(150000)

    def run_hierarchical_cluster(self, params):
        """
        run_hierarchical_cluster: generates hierarchical clusters for Matrix data object

        matrix_ref: Matrix object reference
        workspace_name: the name of the workspace
        cluster_set_name: KBaseExperiments.ClusterSet object name
        dist_cutoff_rate: the threshold to apply when forming flat clusters

        Optional arguments:
        dist_metric: The distance metric to use. Default set to 'euclidean'.
                     The distance function can be
                     ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine",
                      "dice", "euclidean", "hamming", "jaccard", "kulsinski", "matching",
                      "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath",
                      "sqeuclidean", "yule"]
                     Details refer to:
                     https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

        linkage_method: The linkage algorithm to use. Default set to 'single'.
                        The method can be
                        ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
                        Details refer to:
                        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

        fcluster_criterion: The criterion to use in forming flat clusters.
                            Default set to 'inconsistent'.
                            The criterion can be
                            ["inconsistent", "distance", "maxclust"]
                            Details refer to:
                            https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html

        return:
        cluster_set_refs: KBaseExperiments.ClusterSet object references
        report_name: report name generated by KBaseReport
        report_ref: report reference generated by KBaseReport
        """
        logging.info('--->\nrunning run_hierarchical_cluster\n' +
                     'params:\n{}'.format(json.dumps(params, indent=1)))

        self._validate_run_hierarchical_cluster_params(params)

        matrix_ref = params.get('matrix_ref')
        workspace_name = params.get('workspace_name')
        cluster_set_name = params.get('cluster_set_name')
        row_dist_cutoff_rate = float(params.get('row_dist_cutoff_rate', 0.5))
        col_dist_cutoff_rate = float(params.get('col_dist_cutoff_rate', 0.5))
        dist_metric = params.get('dist_metric')
        linkage_method = params.get('linkage_method')
        fcluster_criterion = params.get('fcluster_criterion')

        matrix_object = self.ws.get_objects2({'objects': [{'ref':
                                                          matrix_ref}]})['data'][0]
        matrix_data = matrix_object['data']

        data_matrix = self.gen_api.fetch_data({'obj_ref': matrix_ref}).get('data_matrix')
        transpose_data_matrix = pd.read_json(data_matrix).T.to_json()

        try:
            plotly_heatmap = self._build_plotly_clustermap(data_matrix, dist_metric, linkage_method)
            # plotly_heatmap = self._build_clustermap(data_matrix, dist_metric, linkage_method)
        except:
            plotly_heatmap = None

        (row_flat_cluster,
         row_labels,
         row_newick,
         row_dendrogram_path,
         row_dendrogram_truncate_path) = self._build_flat_cluster(
                                                            data_matrix,
                                                            row_dist_cutoff_rate,
                                                            dist_metric=dist_metric,
                                                            linkage_method=linkage_method,
                                                            fcluster_criterion=fcluster_criterion)

        (col_flat_cluster,
         col_labels,
         col_newick,
         col_dendrogram_path,
         col_dendrogram_truncate_path) = self._build_flat_cluster(
                                                            transpose_data_matrix,
                                                            col_dist_cutoff_rate,
                                                            dist_metric=dist_metric,
                                                            linkage_method=linkage_method,
                                                            fcluster_criterion=fcluster_criterion)

        genome_ref = matrix_data.get('genome_ref')

        clustering_parameters = {'col_dist_cutoff_rate': str(col_dist_cutoff_rate),
                                 'row_dist_cutoff_rate': str(row_dist_cutoff_rate),
                                 'dist_metric': dist_metric,
                                 'linkage_method': linkage_method,
                                 'fcluster_criterion': fcluster_criterion}

        cluster_set_refs = []

        row_cluster_set_name = cluster_set_name + '_row'
        row_cluster_set = self._build_hierarchical_cluster_set(
                                                    row_flat_cluster,
                                                    row_cluster_set_name,
                                                    genome_ref,
                                                    matrix_ref,
                                                    matrix_data.get('row_mapping'),
                                                    matrix_data.get('row_conditionset_ref'),
                                                    workspace_name,
                                                    clustering_parameters,
                                                    data_matrix)
        cluster_set_refs.append(row_cluster_set)

        col_cluster_set_name = cluster_set_name + '_column'
        col_cluster_set = self._build_hierarchical_cluster_set(
                                                    col_flat_cluster,
                                                    col_cluster_set_name,
                                                    genome_ref,
                                                    matrix_ref,
                                                    matrix_data.get('col_mapping'),
                                                    matrix_data.get('col_conditionset_ref'),
                                                    workspace_name,
                                                    clustering_parameters,
                                                    transpose_data_matrix)
        cluster_set_refs.append(col_cluster_set)

        returnVal = {'cluster_set_refs': cluster_set_refs}

        report_output = self._generate_hierarchical_cluster_report(cluster_set_refs,
                                                                   workspace_name,
                                                                   row_dendrogram_path,
                                                                   row_dendrogram_truncate_path,
                                                                   col_dendrogram_path,
                                                                   col_dendrogram_truncate_path,
                                                                   plotly_heatmap)
        returnVal.update(report_output)

        return returnVal
