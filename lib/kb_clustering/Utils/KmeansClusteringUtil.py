
import json
import logging
import uuid
import sys
import errno
import os
import pandas as pd
import seaborn as sns
import traceback
import shutil
import scipy.cluster.vq as vq
from matplotlib import pyplot as plt
from plotly.offline import plot
import plotly.express as px
import plotly.graph_objs as go
from sklearn.decomposition import PCA

from installed_clients.DataFileUtilClient import DataFileUtil
from installed_clients.KBaseReportClient import KBaseReport


class KmeansClusteringUtil:

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

    def _validate_run_kmeans_cluster_params(self, params):
        """
        _validate_run_kmeans_cluster_params:
                validates params passed to run_kmeans_cluster method
        """

        logging.info('start validating run_kmeans_cluster params')

        # check for required parameters
        for p in ['matrix_ref', 'workspace_name', 'cluster_set_name', 'k_num']:
            if p not in params:
                raise ValueError('"{}" parameter is required, but missing'.format(p))

    def _gen_clusters(self, clusters, conditionset_mapping):
        clusters_list = list()

        for cluster in list(clusters.values()):
            labeled_cluster = {}
            labeled_cluster.update({'id_to_data_position': cluster})
            if conditionset_mapping:
                id_to_condition = {k: v for k, v in list(conditionset_mapping.items()) if k in list(cluster.keys())}
                labeled_cluster.update({'id_to_condition': id_to_condition})

            clusters_list.append(labeled_cluster)

        return clusters_list

    def _build_kmeans_cluster_set(self, clusters, cluster_set_name, genome_ref, matrix_ref,
                                  conditionset_mapping, conditionset_ref, workspace_name,
                                  clustering_parameters):
        """
        _build_kmeans_cluster_set: build KBaseExperiments.ClusterSet object
        """

        logging.info('start saving KBaseExperiments.ClusterSet object')

        if isinstance(workspace_name, int) or workspace_name.isdigit():
            workspace_id = workspace_name
        else:
            workspace_id = self.dfu.ws_name_to_id(workspace_name)

        clusters_list = self._gen_clusters(clusters, conditionset_mapping)

        cluster_set_data = {'clusters': clusters_list,
                            'clustering_parameters': clustering_parameters,
                            'original_data': matrix_ref,
                            'condition_set_ref': conditionset_ref,
                            'genome_ref': genome_ref}

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

    def _build_kmeans_cluster(self, data_matrix_df, k_num):
        """
        _build_kmeans_cluster: Build Kmeans cluster
        """

        logging.info('start building clusters')

        data_matrix_df.fillna(0, inplace=True)
        values = data_matrix_df.values
        rows = data_matrix_df.index

        # normalize observations
        logging.info('start normalizing raw data')
        whiten_values = vq.whiten(values)

        # run kmeans algorithm
        logging.info('start performing Kmeans algorithm')
        centroid, idx = vq.kmeans2(whiten_values, k_num, minit='points')

        clusters = {}
        for list_index, value in enumerate(idx):
            cluster = clusters.get(value)
            if not cluster:
                clusters.update({value: {rows[list_index]: list_index}})
            else:
                cluster.update({rows[list_index]: list_index})

        return clusters, idx

    def _generate_pairplot(self, data_df, cluster_labels):

        output_directory = os.path.join(self.scratch, str(uuid.uuid4()))
        self._mkdir_p(output_directory)

        col = data_df.columns

        if len(col) > 20:
            logging.info('start building PCA plot')
            pacplot_path = os.path.join(output_directory, 'pcaplot.html')
            n_components = 2
            s_values = data_df.values
            pca = PCA(n_components=n_components, whiten=True)
            principalComponents = pca.fit_transform(s_values)

            col = list()
            for i in range(n_components):
                col.append('principal_component_{}'.format(i+1))

            rotation_matrix_df = pd.DataFrame(data=principalComponents,
                                              columns=col,
                                              index=data_df.index)
            rotation_matrix_df['cluster'] = [str(cluster_label) for cluster_label in cluster_labels]

            traces = list()
            for name in set(rotation_matrix_df.cluster):
                trace = go.Scatter(
                    x=list(rotation_matrix_df.loc[rotation_matrix_df['cluster'].eq(name)]['principal_component_1']),
                    y=list(rotation_matrix_df.loc[rotation_matrix_df['cluster'].eq(name)]['principal_component_2']),
                    mode='markers',
                    name=name,
                    text=list(rotation_matrix_df.loc[rotation_matrix_df['cluster'].eq(name)].index),
                    textposition='bottom center',
                    marker=go.Marker(size=10, opacity=0.8, line=go.Line(color='rgba(217, 217, 217, 0.14)',
                                                                        width=0.5)))
                traces.append(trace)

            data = go.Data(traces)
            layout = go.Layout(xaxis=go.XAxis(title='Principal Component 1', showline=False),
                               yaxis=go.YAxis(title='Principal Component 2', showline=False))
            fig = go.Figure(data=data, layout=layout)
            fig.update_layout(legend_title='<b> Cluster </b>')

            plot(fig, filename=pacplot_path)
            return pacplot_path

        data_df['cluster'] = [str(cluster_label) for cluster_label in cluster_labels]
        try:
            logging.info('start building pairplot')
            pairplot_path = os.path.join(output_directory, 'pairplot.html')
            fig = px.scatter_matrix(data_df, dimensions=list(col), color='cluster',
                                    symbol='cluster')
            plot(fig, filename=pairplot_path)
        except Exception:
            logging.warning('failed to run plotly pairplot')
            logging.warning(traceback.format_exc())
            logging.warning(sys.exc_info()[2])
            pairplot_path = None

        if not pairplot_path:
            try:
                pairplot_path = os.path.join(output_directory, 'pairplot.png')
                sns_plot = sns.pairplot(data_df, hue='cluster', height=2.5, vars=list(col))
                sns_plot.savefig(pairplot_path)
            except Exception:
                logging.warning('failed to run seaborn pairplot')
                logging.warning(traceback.format_exc())
                logging.warning(sys.exc_info()[2])
                pairplot_path = None

        return pairplot_path

    def _generate_pairplot_content(self, pairplot_path, output_directory,
                                   col=False):

        pairplot_content = ''''''

        if pairplot_path:
            prefix = 'col_' if col else 'row_'
            pairplot_name = prefix + os.path.basename(pairplot_path)
            shutil.copy2(pairplot_path,
                         os.path.join(output_directory, pairplot_name))

            if pairplot_name.endswith('.html'):
                pairplot_content += '''<iframe height="900px" width="100%" '''
                pairplot_content += '''src="{}" style="border:none;"></iframe>\n'''.format(
                                                                                    pairplot_name)
            elif pairplot_name.endswith('.png'):
                pairplot_content += '''\n<img src="{}" '''.format(pairplot_name)
                pairplot_content += '''alt="pairplot" width="480" height="480">\n'''
            else:
                raise ValueError('Unexpected cluster pairplot file format')
        else:
            pairplot_content += '''\n<p style="color:red;" >'''
            pairplot_content += '''Pairplot is too large to be displayed.</p>\n'''

        return pairplot_content

    def _generate_cluster_info_content(self, row_cluster_labels, col_cluster_lables):
        cluster_info = ''''''
        cluster_info += '''\n<p>Row Cluster Info</p>
                                <table style="width:30%">
                                  <tr>
                                    <th>Cluster Index</th>
                                    <th>Size</th>
                                  </tr>\n'''

        unique_labels = list(set(row_cluster_labels))
        unique_labels.sort(key=float)
        for index in unique_labels:
            cluster_info += '''\n<tr>
                                    <td>{}</td>
                                    <td>{}</td>
                            </tr>\n'''.format(index, row_cluster_labels.tolist().count(index))

        cluster_info += '''\n</table>\n'''

        cluster_info += '''\n<br><br>\n'''

        cluster_info += '''\n<p>Column Cluster Info</p>
                                <table style="width:30%">
                                  <tr>
                                    <th>Cluster Index</th>
                                    <th>Size</th>
                                  </tr>\n'''

        unique_labels = list(set(col_cluster_lables))
        unique_labels.sort(key=float)
        for index in unique_labels:
            cluster_info += '''\n<tr>
                                    <td>{}</td>
                                    <td>{}</td>
                            </tr>\n'''.format(index, col_cluster_lables.tolist().count(index))

        cluster_info += '''\n</table>\n'''

        return cluster_info

    def _generate_kmeans_html_report(self, data_matrix_df, row_cluster_labels,
                                     transpose_data_matrix_df, col_cluster_lables):

        logging.info('start generating html report')
        html_report = list()

        output_directory = os.path.join(self.scratch, str(uuid.uuid4()))
        self._mkdir_p(output_directory)
        result_file_path = os.path.join(output_directory, 'hier_report.html')

        row_pairplot = self._generate_pairplot(data_matrix_df, row_cluster_labels)
        col_pairplot = self._generate_pairplot(transpose_data_matrix_df, col_cluster_lables)

        cluster_info = self._generate_cluster_info_content(row_cluster_labels, col_cluster_lables)
        row_pairplot_content = self._generate_pairplot_content(row_pairplot, output_directory)
        col_pairplot_content = self._generate_pairplot_content(col_pairplot, output_directory,
                                                               col=True)

        with open(result_file_path, 'w') as result_file:
            with open(os.path.join(os.path.dirname(__file__), 'kmeans_report_template.html'),
                      'r') as report_template_file:
                report_template = report_template_file.read()
                report_template = report_template.replace('<p>Cluster_Info</p>',
                                                          cluster_info)
                report_template = report_template.replace('<p>Row_Pairplot</p>',
                                                          row_pairplot_content)
                report_template = report_template.replace('<p>Column_Pairplot</p>',
                                                          col_pairplot_content)
                result_file.write(report_template)

        report_shock_id = self.dfu.file_to_shock({'file_path': output_directory,
                                                  'pack': 'zip'})['shock_id']

        html_report.append({'shock_id': report_shock_id,
                            'name': os.path.basename(result_file_path),
                            'label': os.path.basename(result_file_path),
                            'description': 'HTML summary report for Cluster App'
                            })
        return html_report

    def _generate_kmeans_cluster_report(self, cluster_set_refs, workspace_name,
                                        data_matrix_df, row_cluster_labels,
                                        transpose_data_matrix_df, col_cluster_lables):
        """
        _generate_kmeans_cluster_report: generate summary report
        """
        objects_created = []
        for cluster_set_ref in cluster_set_refs:
            objects_created.append({'ref': cluster_set_ref,
                                    'description': 'Kmeans ClusterSet'})

        output_html_files = self._generate_kmeans_html_report(data_matrix_df,
                                                              row_cluster_labels,
                                                              transpose_data_matrix_df,
                                                              col_cluster_lables)

        report_params = {'message': '',
                         'workspace_name': workspace_name,
                         'objects_created': objects_created,
                         'html_links': output_html_files,
                         'direct_html_link_index': 0,
                         'html_window_height': 333,
                         'report_object_name': 'run_kmeans_cluster_' + str(uuid.uuid4())}

        kbase_report_client = KBaseReport(self.callback_url, token=self.token)
        output = kbase_report_client.create_extended_report(report_params)

        report_output = {'report_name': output['name'], 'report_ref': output['ref']}

        return report_output

    def __init__(self, config):
        self.callback_url = config['SDK_CALLBACK_URL']
        self.token = config['KB_AUTH_TOKEN']
        self.scratch = config['scratch']

        # helper kbase module
        self.dfu = DataFileUtil(self.callback_url)

        logging.basicConfig(format='%(created)s %(levelname)s: %(message)s',
                            level=logging.INFO)

        plt.switch_backend('agg')
        sys.setrecursionlimit(150000)

    def run_kmeans_cluster(self, params):
        """
        run_kmeans_cluster: generates Kmeans clusters for Matrix data object

        matrix_ref: Matrix object reference
        workspace_name: the name of the workspace
        cluster_set_name: KBaseExperiments.ClusterSet object name
        k_num: number of clusters to form

        Optional arguments:
        dist_metric: The distance metric to use. Default set to 'euclidean'.
                     The distance function can be
                     ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine",
                      "dice", "euclidean", "hamming", "jaccard", "kulsinski", "matching",
                      "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath", "sqeuclidean",
                      "yule"]
                     Details refer to:
                     https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

        return:
        cluster_set_refs: KBaseExperiments.ClusterSet object references
        report_name: report name generated by KBaseReport
        report_ref: report reference generated by KBaseReport
        """

        logging.info('--->\nrunning run_kmeans_cluster\n' +
                     'params:\n{}'.format(json.dumps(params, indent=1)))

        self._validate_run_kmeans_cluster_params(params)

        matrix_ref = params.get('matrix_ref')
        workspace_name = params.get('workspace_name')
        cluster_set_name = params.get('cluster_set_name')
        k_num = params.get('k_num')

        matrix_data = self.dfu.get_objects({'object_refs': [matrix_ref]})['data'][0]['data']

        matrix_data_values = matrix_data['data']
        data_matrix_df = pd.DataFrame(matrix_data_values['values'],
                                      index=matrix_data_values['row_ids'],
                                      columns=matrix_data_values['col_ids'])
        transpose_data_matrix_df = data_matrix_df.T

        (row_kmeans_clusters,
         row_cluster_labels) = self._build_kmeans_cluster(data_matrix_df, k_num)

        (col_kmeans_clusters,
         col_cluster_lables) = self._build_kmeans_cluster(transpose_data_matrix_df, k_num)

        genome_ref = matrix_data.get('genome_ref')
        clustering_parameters = {'k_num': str(k_num)}

        cluster_set_refs = []

        row_cluster_set_name = cluster_set_name + '_row'
        row_cluster_set = self._build_kmeans_cluster_set(
                                                    row_kmeans_clusters,
                                                    row_cluster_set_name,
                                                    genome_ref,
                                                    matrix_ref,
                                                    matrix_data.get('row_mapping'),
                                                    matrix_data.get('row_conditionset_ref'),
                                                    workspace_name,
                                                    clustering_parameters)
        cluster_set_refs.append(row_cluster_set)

        col_cluster_set_name = cluster_set_name + '_column'
        col_cluster_set = self._build_kmeans_cluster_set(
                                                    col_kmeans_clusters,
                                                    col_cluster_set_name,
                                                    genome_ref,
                                                    matrix_ref,
                                                    matrix_data.get('col_mapping'),
                                                    matrix_data.get('col_conditionset_ref'),
                                                    workspace_name,
                                                    clustering_parameters)
        cluster_set_refs.append(col_cluster_set)

        returnVal = {'cluster_set_refs': cluster_set_refs}

        report_output = self._generate_kmeans_cluster_report(cluster_set_refs, workspace_name,
                                                             data_matrix_df,
                                                             row_cluster_labels,
                                                             transpose_data_matrix_df,
                                                             col_cluster_lables)

        returnVal.update(report_output)

        return returnVal
