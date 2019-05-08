# -*- coding: utf-8 -*-
import os
import time
import unittest
from configparser import ConfigParser
import shutil
import inspect

from kb_clustering.kb_clusteringImpl import kb_clustering
from kb_clustering.kb_clusteringServer import MethodContext
from kb_clustering.authclient import KBaseAuth as _KBaseAuth

from installed_clients.WorkspaceClient import Workspace
from installed_clients.DataFileUtilClient import DataFileUtil
from installed_clients.GenericsAPIClient import GenericsAPI
from installed_clients.GenomeFileUtilClient import GenomeFileUtil


class kb_clusteringTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        token = os.environ.get('KB_AUTH_TOKEN', None)
        config_file = os.environ.get('KB_DEPLOYMENT_CONFIG', None)
        cls.cfg = {}
        config = ConfigParser()
        config.read(config_file)
        for nameval in config.items('kb_clustering'):
            cls.cfg[nameval[0]] = nameval[1]
        # Getting username from Auth profile for token
        authServiceUrl = cls.cfg['auth-service-url']
        auth_client = _KBaseAuth(authServiceUrl)
        user_id = auth_client.get_user(token)
        # WARNING: don't call any logging methods on the context object,
        # it'll result in a NoneType error
        cls.ctx = MethodContext(None)
        cls.ctx.update({'token': token,
                        'user_id': user_id,
                        'provenance': [
                            {'service': 'kb_clustering',
                             'method': 'please_never_use_it_in_production',
                             'method_params': []
                             }],
                        'authenticated': 1})
        cls.wsURL = cls.cfg['workspace-url']
        cls.wsClient = Workspace(cls.wsURL)
        cls.serviceImpl = kb_clustering(cls.cfg)
        cls.scratch = cls.cfg['scratch']
        cls.callback_url = os.environ['SDK_CALLBACK_URL']
        suffix = int(time.time() * 1000)
        cls.wsName = "test_ContigFilter_" + str(suffix)
        ret = cls.wsClient.create_workspace({'workspace': cls.wsName})  # noqa

        cls.gfu = GenomeFileUtil(cls.callback_url)
        cls.dfu = DataFileUtil(cls.callback_url)
        cls.gen_api = GenericsAPI(cls.callback_url, service_ver='dev')

        cls.prepare_data()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'wsName'):
            cls.wsClient.delete_workspace({'workspace': cls.wsName})
            print('Test workspace was deleted')

    def getImpl(self):
        return self.__class__.serviceImpl

    def getWsName(self):
        return self.__class__.wsName

    @classmethod
    def prepare_data(cls):
        # upload Genome object
        genbank_file_name = 'minimal.gbff'
        genbank_file_path = os.path.join(cls.scratch, genbank_file_name)
        shutil.copy(os.path.join('data', genbank_file_name), genbank_file_path)

        genome_object_name = 'test_Genome'
        cls.genome_ref = cls.gfu.genbank_to_genome({'file': {'path': genbank_file_path},
                                                    'workspace_name': cls.wsName,
                                                    'genome_name': genome_object_name,
                                                    'generate_ids_if_needed': 1
                                                    })['genome_ref']

        # upload KBaseFeatureValues.ExpressionMatrix object
        workspace_id = cls.dfu.ws_name_to_id(cls.wsName)
        object_type = 'KBaseFeatureValues.ExpressionMatrix'
        expression_matrix_object_name = 'test_expression_matrix'
        expression_matrix_data = {'genome_ref': cls.genome_ref,
                                  'scale': 'log2',
                                  'type': 'level',
                                  'data': {'row_ids': ['gene_1', 'gene_2', 'gene_3'],
                                           'col_ids': ['condition_1', 'condition_2',
                                                       'condition_3', 'condition_4'],
                                           'values': [[0.1, 0.2, 0.3, 0.4],
                                                      [0.3, 0.4, 0.5, 0.6],
                                                      [None, None, None, None]]
                                           },
                                  'feature_mapping': {},
                                  'condition_mapping': {}}
        save_object_params = {
            'id': workspace_id,
            'objects': [{'type': object_type,
                         'data': expression_matrix_data,
                         'name': expression_matrix_object_name}]
        }

        dfu_oi = cls.dfu.save_objects(save_object_params)[0]
        cls.expression_matrix_ref = str(dfu_oi[6]) + '/' + str(dfu_oi[0]) + '/' + str(dfu_oi[4])

        # upload KBaseMatrices.ExpressionMatrix object
        matrix_file_name = 'test_import.xlsx'
        matrix_file_path = os.path.join(cls.scratch, matrix_file_name)
        shutil.copy(os.path.join('data', matrix_file_name), matrix_file_path)

        obj_type = 'ExpressionMatrix'
        params = {'obj_type': obj_type,
                  'matrix_name': 'test_ExpressionMatrix',
                  'workspace_name': cls.wsName,
                  'input_file_path': matrix_file_path,
                  'genome_ref': cls.genome_ref,
                  'scale': 'raw'}
        gen_api_ret = cls.gen_api.import_matrix_from_excel(params)

        cls.matrix_obj_ref = gen_api_ret.get('matrix_obj_ref')
        matrix_obj_data = cls.dfu.get_objects(
            {"object_refs": [cls.matrix_obj_ref]})['data'][0]['data']

        cls.col_conditionset_ref = matrix_obj_data.get('col_attributemapping_ref')
        cls.row_conditionset_ref = matrix_obj_data.get('row_attributemapping_ref')

    def fail_run_kmeans_cluster(self, params, error, exception=ValueError,
                                contains=False):
        with self.assertRaises(exception) as context:
            self.getImpl().run_kmeans_cluster(self.ctx, params)
        if contains:
            self.assertIn(error, str(context.exception.args[0]))
        else:
            self.assertEqual(error, str(context.exception.args[0]))

    def check_run_kmeans_cluster_output(self, ret):
        self.assertTrue('cluster_set_refs' in ret)
        self.assertTrue('report_name' in ret)
        self.assertTrue('report_ref' in ret)

    def start_test(self):
        testname = inspect.stack()[1][3]
        print(('\n*** starting test: ' + testname + ' **'))

    def test_bad_run_kmeans_cluster_params(self):
        self.start_test()
        invalidate_params = {'missing_matrix_ref': 'matrix_ref',
                             'workspace_name': 'workspace_name',
                             'cluster_set_name': 'cluster_set_name',
                             'k_num': 'k_num'}
        error_msg = '"matrix_ref" parameter is required, but missing'
        self.fail_run_kmeans_cluster(invalidate_params, error_msg)

        invalidate_params = {'matrix_ref': 'matrix_ref',
                             'workspace_name': 'workspace_name',
                             'cluster_set_name': 'cluster_set_name',
                             'k_num': 'k_num',
                             'dist_metric': 'invalidate_metric'}
        error_msg = 'INPUT ERROR:\nInput metric function [invalidate_metric] is not valid.\n'
        self.fail_run_kmeans_cluster(invalidate_params, error_msg, contains=True)

    def test_run_kmeans_cluster(self):
        self.start_test()

        params = {'matrix_ref': self.matrix_obj_ref,
                  'workspace_name': self.getWsName(),
                  'cluster_set_name': 'test_kmeans_cluster',
                  'k_num': 2,
                  'dist_metric': 'cityblock'}
        ret = self.getImpl().run_kmeans_cluster(self.ctx, params)[0]
        self.check_run_kmeans_cluster_output(ret)
