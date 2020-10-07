
import json
import os
import logging
import errno
import uuid
import sys
import pandas as pd
import shutil
import traceback
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import plot
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist


from installed_clients.DataFileUtilClient import DataFileUtil
from installed_clients.KBaseReportClient import KBaseReport


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
        for p in ['matrix_ref', 'workspace_id', 'cluster_set_name']:
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

    def _build_plotly_clustermap(self, df, dist_metric, linkage_method):

        logging.info('start building plotly page')

        output_directory = os.path.join(self.scratch, str(uuid.uuid4()))
        self._mkdir_p(output_directory)
        plot_file = os.path.join(output_directory, 'clustermap.html')

        df.fillna(0, inplace=True)

        # Initialize figure by creating upper dendrogram
        logging.info('initializing upper dendrogram')
        figure = ff.create_dendrogram(df.T, orientation='bottom', labels=df.T.index,
                                      linkagefun=lambda x: hier.linkage(df.T.values,
                                                                        method=linkage_method,
                                                                        metric=dist_metric))
        for i in range(len(figure['data'])):
            figure['data'][i]['yaxis'] = 'y2'

        # Create Side Dendrogram
        logging.info('creating side dendrogram')
        dendro_side = ff.create_dendrogram(df, orientation='right', labels=df.index,
                                           linkagefun=lambda x: hier.linkage(
                                                                        df.values,
                                                                        method=linkage_method,
                                                                        metric=dist_metric))
        for i in range(len(dendro_side['data'])):
            dendro_side['data'][i]['xaxis'] = 'x2'

        # Add Side Dendrogram Data to Figure
        figure.add_traces(dendro_side['data'])
        # figure['data'].extend(dendro_side['data'])

        # Create Heatmap
        logging.info('creating heatmap')
        heatmap = [go.Heatmap(x=df.columns, y=df.index, z=df.values, colorscale='YlGnBu')]

        original_heatmap_x = heatmap[0]['x']
        original_heatmap_y = heatmap[0]['y']

        heatmap[0]['x'] = figure['layout']['xaxis']['tickvals']
        heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

        # Add Heatmap Data to Figure
        figure.add_traces(heatmap)
        # figure['data'].extend(heatmap)

        # Edit Layout
        figure['layout'].update({'width': 800, 'height': 800,
                                 'showlegend': False, 'hovermode': 'closest',
                                 })
        # Edit xaxis
        figure['layout']['xaxis'].update({'domain': [.15, 1],
                                          'mirror': False,
                                          'showgrid': False,
                                          'showline': False,
                                          'zeroline': False,
                                          'ticktext': original_heatmap_x,
                                          'ticks': ""})
        # Edit xaxis2
        figure['layout'].update({'xaxis2': {'domain': [0, .15],
                                            'mirror': False,
                                            'showgrid': False,
                                            'showline': False,
                                            'zeroline': False,
                                            'showticklabels': False,
                                            'ticktext': original_heatmap_x,
                                            'ticks': ""}})

        # Edit yaxis
        figure['layout']['yaxis'] = dendro_side['layout']['yaxis']
        figure['layout']['yaxis'].update({'domain': [0, .85],
                                          'mirror': False,
                                          'showgrid': False,
                                          'showline': False,
                                          'zeroline': False,
                                          'showticklabels': False,
                                          'ticktext': original_heatmap_y,
                                          'ticks': ""})
        # Edit yaxis2
        figure['layout'].update({'yaxis2': {'domain': [.825, .975],
                                            'mirror': False,
                                            'showgrid': False,
                                            'showline': False,
                                            'zeroline': False,
                                            'showticklabels': False,
                                            'ticks': ""}})

        logging.info('plotting figure')
        plot(figure, filename=plot_file)

        return plot_file

    def _generate_visualization_content(self, output_directory, clusterheatmap):

        """
        _generate_visualization_content: generate visualization html content
        """

        clusterheatmap_content = ''

        if clusterheatmap is None:
            clusterheatmap_content += 'clustet heatmap is too large to display'
        elif os.path.basename(clusterheatmap).endswith('.html'):
            clusterheatmap_html = 'clusterheatmap.html'
            shutil.copy2(clusterheatmap,
                         os.path.join(output_directory, clusterheatmap_html))

            clusterheatmap_content += '<iframe height="900px" width="100%" '
            clusterheatmap_content += 'src="{}" style="border:none;"></iframe>'.format(clusterheatmap_html)
        else:
            raise ValueError('Unexpected cluster heatmap file format')

        return clusterheatmap_content

    def _generate_cluster_info_content(self, flat_cluster):

        cluster_info = ''''''
        cluster_info += '''\n<br><br>
                                <table style="width:30%">
                                  <tr>
                                    <th>Cluster Index</th>
                                    <th>Size</th>
                                  </tr>\n'''

        keys = list(flat_cluster.keys())
        keys.sort(key=float)
        for index in keys:
            labels = flat_cluster[index]
            cluster_info += '''\n<tr>
                                    <td>{}</td>
                                    <td>{}</td>
                            </tr>\n'''.format(index, len(labels))

        cluster_info += '''\n</table>\n'''

        return cluster_info

    def _generate_dendrogram_content(self, dendrogram_path, truncated, output_directory,
                                     dimension):

        dendrogramo_content = ''''''

        if dendrogram_path:
            dendrogram_name = dimension + '_' + os.path.basename(dendrogram_path)
            shutil.copy2(dendrogram_path,
                         os.path.join(output_directory, dendrogram_name))

            if dendrogram_name.endswith('.html'):
                dendrogramo_content += '''<iframe height="900px" width="100%" '''
                dendrogramo_content += '''src="{}" style="border:none;"></iframe>\n'''.format(
                                                                                dendrogram_name)
            elif dendrogram_name.endswith('.png'):
                if truncated:
                    dendrogramo_content += '''\n<p style="color:red;" >'''
                    dendrogramo_content += 'Full dendrogram is too large to be displayed.</p>\n'
                    dendrogramo_content += '''\n<p style="color:red;" >'''
                    dendrogramo_content += '''Showing the last 12 merged clusters.</p>\n'''
                dendrogramo_content += '''\n<img src="{}" '''.format(dendrogram_name)
                dendrogramo_content += '''alt="dendrogram" width="460" height="345">\n'''
            else:
                raise ValueError('Unexpected cluster dendrogram file format')
        else:
            dendrogramo_content += '''\n<p style="color:red;" >'''
            dendrogramo_content += '''Dendrogram is too large to be displayed.</p>\n'''

        return dendrogramo_content

    def _generate_hierarchical_html_report(self, flat_cluster, dendrogram_path, truncated,
                                           dimension):
        """
        _generate_hierarchical_html_report: generate html summary report for hierarchical
                                            clustering app
        """

        logging.info('start generating html report')
        html_report = list()

        output_directory = os.path.join(self.scratch, str(uuid.uuid4()))
        self._mkdir_p(output_directory)
        result_file_path = os.path.join(output_directory, 'hier_report.html')

        cluster_info = self._generate_cluster_info_content(flat_cluster)

        dendrogramo_content = self._generate_dendrogram_content(dendrogram_path,
                                                                truncated,
                                                                output_directory,
                                                                dimension)

        with open(result_file_path, 'w') as result_file:
            with open(os.path.join(os.path.dirname(__file__), 'hier_report_template.html'),
                      'r') as report_template_file:
                report_template = report_template_file.read()
                report_template = report_template.replace('<p>Cluster_Info</p>',
                                                          cluster_info)
                report_template = report_template.replace('<p>Dendrogram</p>',
                                                          dendrogramo_content)
                result_file.write(report_template)

        report_shock_id = self.dfu.file_to_shock({'file_path': output_directory,
                                                  'pack': 'zip'})['shock_id']

        html_report.append({'shock_id': report_shock_id,
                            'name': os.path.basename(result_file_path),
                            'label': os.path.basename(result_file_path),
                            'description': 'HTML summary report for Cluster App'
                            })
        return html_report

    def _generate_hierarchical_cluster_report(self,  cluster_set_ref, workspace_id,
                                              flat_cluster, dendrogram_path, truncated, dimension):
        """
        _generate_hierarchical_cluster_report: generate summary report
        """

        objects_created = []
        objects_created.append({'ref': cluster_set_ref,
                                'description': 'Hierarchical ClusterSet'})

        output_html_files = self._generate_hierarchical_html_report(flat_cluster,
                                                                    dendrogram_path,
                                                                    truncated,
                                                                    dimension)

        report_params = {'message': '',
                         'workspace_id': workspace_id,
                         'objects_created': objects_created,
                         'html_links': output_html_files,
                         'direct_html_link_index': 0,
                         'html_window_height': 333,
                         'report_object_name': 'run_hierarchical_cluster_' + str(uuid.uuid4())}

        kbase_report_client = KBaseReport(self.callback_url, token=self.token)
        output = kbase_report_client.create_extended_report(report_params)

        report_output = {'report_name': output['name'], 'report_ref': output['ref']}

        return report_output

    def _gen_hierarchical_clusters(self, clusters, conditionset_mapping, data_matrix_df):
        clusters_list = list()

        index = data_matrix_df.index.tolist()

        for cluster in list(clusters.values()):
            labeled_cluster = {}
            id_to_data_position = {}
            for item in cluster:
                id_to_data_position.update({item: index.index(item)})

            labeled_cluster.update({'id_to_data_position': id_to_data_position})
            if conditionset_mapping:
                id_to_condition = {k: v for k, v in list(conditionset_mapping.items()) if k in cluster}
                labeled_cluster.update({'id_to_condition': id_to_condition})

            clusters_list.append(labeled_cluster)

        return clusters_list

    def _build_hierarchical_cluster_set(self, clusters, cluster_set_name, matrix_ref,
                                        conditionset_mapping, conditionset_ref, workspace_id,
                                        clustering_parameters, data_matrix_df):

        """
        _build_hierarchical_cluster_set: build KBaseExperiments.ClusterSet object
        """

        logging.info('start saving KBaseExperiments.ClusterSet object')

        clusters_list = self._gen_hierarchical_clusters(clusters, conditionset_mapping,
                                                        data_matrix_df)

        cluster_set_data = {'clusters': clusters_list,
                            'clustering_parameters': clustering_parameters,
                            'original_data': matrix_ref,
                            'condition_set_ref': conditionset_ref}

        cluster_set_data = {k: v for k, v in list(cluster_set_data.items()) if v}

        object_type = 'KBaseExperiments.ClusterSet'
        save_object_params = {
            'id': workspace_id,
            'objects': [{'type': object_type,
                         'data': cluster_set_data,
                         'name': cluster_set_name}]}

        dfu_oi = self.dfu.save_objects(save_object_params)[0]
        cluster_set_ref = str(dfu_oi[6]) + '/' + str(dfu_oi[0]) + '/' + str(dfu_oi[4])

        return cluster_set_ref

    def _process_fcluster(self, fcluster, labels):
        """
        _process_fcluster: assign labels to corresponding cluster group
                           if labels is none, return element pos array in each cluster group
        """

        logging.info('start assigning labels to clusters')

        flat_cluster = {}
        for pos, element in enumerate(fcluster):
            cluster_name = str(element)
            if cluster_name not in flat_cluster:
                flat_cluster.update({cluster_name: [labels[pos]]})
            else:
                cluster = flat_cluster.get(cluster_name)
                cluster.append(labels[pos])
                flat_cluster.update({cluster_name: cluster})

        return flat_cluster

    def _add_distance(self, ddata):
        """
        _add_distance: Add distance and cluster count to dendrogram
        credit --
        Author: Jorn Hees
        https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/#Eye-Candy
        """

        logging.info('start adding distance to dendrogram')

        annotate_above = 10  # useful in small plots so annotations don't overlap
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')

    def _build_dendrogram_path(self, linkage_matrix, dist_threshold, labels):
        logging.info('start building dendrogram')

        output_directory = os.path.join(self.scratch, str(uuid.uuid4()))
        self._mkdir_p(output_directory)
        dendrogram_path = os.path.join(output_directory, 'dendrogram.png')

        truncated = False

        if len(linkage_matrix) > 256:
            last_merges = 12
            plt.figure(figsize=(25, 10))
            plt.ylabel('distance')
            plt.title('Hierarchical Clustering Dendrogram (last 12 merges)')
            plt.xlabel('cluster size')
            ddata = hier.dendrogram(linkage_matrix,
                                    leaf_rotation=45.,
                                    leaf_font_size=8.,
                                    show_leaf_counts=True,
                                    labels=labels,
                                    truncate_mode='lastp',
                                    p=last_merges,
                                    show_contracted=True)
            truncated = True
        else:
            figure_width = min(max(len(labels) // 100 * 25, 25), 350)
            plt.figure(figsize=(figure_width, figure_width//25*10))
            plt.ylabel('distance')

            plt.title('Hierarchical Clustering Dendrogram')
            if labels:
                plt.xlabel('labels')
            else:
                plt.xlabel('index')
            ddata = hier.dendrogram(linkage_matrix,
                                    leaf_rotation=45.,
                                    leaf_font_size=8.,
                                    show_leaf_counts=True,
                                    labels=labels)

        self._add_distance(ddata)

        if dist_threshold:
            plt.axhline(y=dist_threshold, c='k')

        plt.savefig(dendrogram_path)

        return dendrogram_path, truncated

    def _build_flat_cluster(self, data_matrix_df, dist_cutoff_rate,
                            dist_metric=None, linkage_method=None, fcluster_criterion=None):
        """
        _build_cluster: build flat clusters and dendrogram for data_matrix
        """

        logging.info('start building clusters')

        data_matrix_df.fillna(0, inplace=True)
        values = data_matrix_df.values
        labels = data_matrix_df.index.tolist()

        # calculate distance matrix
        logging.info('start calculating distance matrix')
        dist_matrix = dist.pdist(values, metric=dist_metric)

        # calculate linkage matrix
        logging.info('start calculating linkage matrix')
        linkage_matrix = hier.linkage(dist_matrix, method=linkage_method)

        height = max([item[2] for item in linkage_matrix])
        dist_threshold = height * dist_cutoff_rate
        logging.info('Height: {} Setting dist_threshold: {}'.format(height, dist_threshold))

        # generate flat clusters
        logging.info('start calculating flat clusters')
        fcluster = hier.fcluster(linkage_matrix, dist_threshold, criterion=fcluster_criterion)
        flat_cluster = self._process_fcluster(fcluster, labels=labels)

        dendrogram_path = None
        truncated = False
        if len(labels) < 60:
            try:
                output_directory = os.path.join(self.scratch, str(uuid.uuid4()))
                self._mkdir_p(output_directory)
                dendrogram_path = os.path.join(output_directory, 'dendrogram.html')

                # Initialize figure by creating upper dendrogram
                logging.info('start initializing dendrogram')
                figure = ff.create_dendrogram(data_matrix_df, orientation='bottom',
                                              labels=labels,
                                              linkagefun=lambda x: hier.linkage(
                                                                            values,
                                                                            method=linkage_method,
                                                                            metric=dist_metric))
                figure.update_layout(xaxis={'automargin': True,
                                            'tickangle': 45})
                logging.info('start plotting figure')
                plot(figure, filename=dendrogram_path)
            except Exception:
                logging.warning('failed to run plotly pairplot')
                logging.warning(traceback.format_exc())
                logging.warning(sys.exc_info()[2])

        if not dendrogram_path:
            try:
                (dendrogram_path,
                 truncated) = self._build_dendrogram_path(linkage_matrix, dist_threshold, labels)
            except Exception:
                logging.warning('failed to run _build_dendrogram_path')
                logging.warning(traceback.format_exc())
                logging.warning(sys.exc_info()[2])
                dendrogram_path = None
                truncated = False

        return flat_cluster, dendrogram_path, truncated

    def __init__(self, config):
        self.callback_url = config['SDK_CALLBACK_URL']
        self.token = config['KB_AUTH_TOKEN']
        self.scratch = config['scratch']

        # helper kbase module
        self.dfu = DataFileUtil(self.callback_url)

        plt.switch_backend('agg')
        sys.setrecursionlimit(150000)

    def run_hierarchical_cluster(self, params):
        """
        run_hierarchical_cluster: generates hierarchical clusters for Matrix data object

        matrix_ref: Matrix object reference
        workspace_id: the id of the workspace
        cluster_set_name: KBaseExperiments.ClusterSet object name
        dimension: run cluster algorithm on dimension, col or row
        dist_cutoff_rate: the threshold to apply when forming flat clusters

        Optional arguments:
        dist_metric: The distance metric to use. Default set to 'euclidean'.
                   The distance function can be
                   ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine",
                    "dice", "euclidean", "hamming", "jaccard", "kulsinski", "matching",
                    "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath", "sqeuclidean",
                    "yule"]
                   Details refer to:
                   https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

        linkage_method: The linkage algorithm to use. Default set to 'ward'.
                      The method can be
                      ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
                      Details refer to:
                      https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

        fcluster_criterion: The criterion to use in forming flat clusters. Default set to 'distance'.
                          The criterion can be
                          ["inconsistent", "distance", "maxclust"]
                          Details refer to:
                          https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html

        return:
        cluster_set_ref: KBaseExperiments.ClusterSet object reference
        report_name: report name generated by KBaseReport
        report_ref: report reference generated by KBaseReport
        """
        logging.info('--->\nrunning run_hierarchical_cluster\n' +
                     'params:\n{}'.format(json.dumps(params, indent=1)))

        self._validate_run_hierarchical_cluster_params(params)

        matrix_ref = params.get('matrix_ref')
        workspace_id = params.get('workspace_id')
        cluster_set_name = params.get('cluster_set_name')
        dimension = params.get('dimension', 'col')
        dist_cutoff_rate = float(params.get('dist_cutoff_rate', 0.5))

        dist_metric = params.get('dist_metric', 'euclidean')
        linkage_method = params.get('linkage_method', 'ward')
        fcluster_criterion = params.get('fcluster_criterion', 'inconsistent')

        input_obj = self.dfu.get_objects({'object_refs': [matrix_ref]})['data'][0]
        input_obj_data = input_obj['data']

        if dimension not in ['col', 'row']:
            raise ValueError('Please use "col" or "row" for input dimension')

        # if "KBaseMatrices" not in obj_type and 'KBaseProfile' not in obj_type:
        #     err_msg = 'Ooops! [{}] is not supported.\n'.format(obj_type)
        #     err_msg += 'Please supply KBaseMatrices or KBaseProfile object'
        #     raise ValueError(err_msg)

        matrix_data_values = input_obj_data['data']
        data_matrix_df = pd.DataFrame(matrix_data_values['values'],
                                      index=matrix_data_values['row_ids'],
                                      columns=matrix_data_values['col_ids'])

        if dimension == 'col':
            data_matrix_df = data_matrix_df.T

        (flat_cluster,
         dendrogram_path,
         truncated) = self._build_flat_cluster(data_matrix_df,
                                               dist_cutoff_rate,
                                               dist_metric=dist_metric,
                                               linkage_method=linkage_method,
                                               fcluster_criterion=fcluster_criterion)

        clustering_parameters = {'dimension': dimension,
                                 'dist_cutoff_rate': str(dist_cutoff_rate),
                                 'dist_metric': dist_metric,
                                 'linkage_method': linkage_method,
                                 'fcluster_criterion': fcluster_criterion}

        cluster_set_ref = self._build_hierarchical_cluster_set(
                                    flat_cluster,
                                    cluster_set_name,
                                    matrix_ref,
                                    input_obj_data.get('{}_mapping'.format(dimension)),
                                    input_obj_data.get('{}_conditionset_ref'.format(dimension)),
                                    workspace_id,
                                    clustering_parameters,
                                    data_matrix_df)
        returnVal = {'cluster_set_ref': cluster_set_ref}

        report_output = self._generate_hierarchical_cluster_report(cluster_set_ref,
                                                                   workspace_id,
                                                                   flat_cluster,
                                                                   dendrogram_path,
                                                                   truncated,
                                                                   dimension)
        returnVal.update(report_output)

        return returnVal
