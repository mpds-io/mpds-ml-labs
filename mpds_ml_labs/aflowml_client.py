"""
Taken from the AFLOW webpage
http://aflow.org/src/aflow-ml
due to absense of the PyPI package
"""
import json
import sys
from time import sleep

# Import proper urllib versions depending on Python version
if sys.version_info >= (3,0):
    from urllib.parse import urlencode
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.error import HTTPError
else:
    from urllib2 import Request
    from urllib2 import urlopen
    from urllib import urlencode
    from urllib2 import HTTPError


class AFLOWmlAPIError(Exception):
    def __init__(self, error_message, status_code=None):
        self.status_code = status_code
        self.error_message = error_message

    def __str__(self):
        if self.status_code:
            return '(%s) %s' % (self.status_code, self.error_message)
        else:
            return self.error_message

def urlencoder(query):
    if sys.version_info >= (3,0):
        return urlencode(query).encode('utf-8')
    else:
        return urlencode(query)

def json_loader(content):
    if sys.version_info >= (3,0):
        return json.loads(content.decode('utf-8'))
    else:
        return json.loads(content)


class AFLOWmlAPI:

    def __init__(self):
        self._base_url = 'http://aflow.org/API/aflow-ml/v1.1'
        self.res_data = {}
        self.model = None
        self.supported_models = [
            'plmf',
            'mfd',
            'asc'
        ]
        self.plmf_fields = [
            'ml_egap_type',
            'ml_egap',
            'ml_energy_per_atom',
            'ml_ael_bulk_modulus_vrh',
            'ml_ael_shear_modulus_vrh',
            'ml_agl_debye',
            'ml_agl_heat_capacity_Cp_300K',
            'ml_agl_heat_capacity_Cp_300K_per_atom',
            'ml_agl_heat_capacity_Cv_300K',
            'ml_agl_heat_capacity_Cv_300K_per_atom',
            'ml_agl_thermal_conductivity_300K',
            'ml_agl_thermal_expansion_300K'
        ]
        self.mfd_fields = [
            'ml_Cv',
            'ml_Fvib',
            'ml_Svib'
        ]

        self.asc_fields = [
            'ml_Tc_5K',
            'ml_Tc_10K'
        ]

    def submit_job(self, post_data, model):
        '''
        Post the contents of post_data to the API endpoint
        <model>/prediction.

        Returns the task id used to poll the job.

        Throws AFLOWmlError if invalid model, HTTPError or invalid response.
        '''
        if model not in self.supported_models:
            raise AFLOWmlAPIError(
                'The model you specified is not valid. Please select from' +
                ' the following: \n' + '\n'.join(
                    ['   ' + s for s in self.supported_models]
                )
            )
        self.model = model
        if model != 'asc':
            encoded_data = urlencoder({
                'file': post_data,
            })
        else:
            encoded_data = urlencoder({
                'composition': post_data
            })
        url = self._base_url + '/' + self.model + '/prediction'
        req = Request(url, encoded_data)
        res = None
        try:
            res = urlopen(req).read()
        except HTTPError as e:
            raise AFLOWmlAPIError(
                'Failed to submit job: {}'.format(e.code)
            )

        res_json = None
        try:
            res_json = json_loader(res)
        except ValueError:
            raise AFLOWmlAPIError(
                'Unable to parse response, invalid JSON'
            )

        self.res_data = {}
        return res_json['id']

    def poll_job(self, job_id, fields=[]):
        '''
        From the job id, polls the API enpoint /prediction/result/<job_id> to
        check the status of the job. Polls until status = SUCCESS.

        Returns prediction object as a dictionary.

        Throws AFLOWmlAPIError if unable to poll job, status = FAILURE,
        HTTPError or invalid response.
        '''
        if fields:
            valid_field = False
            if self.model == 'plmf':
                valid_field = set(fields).issubset(set(self.plmf_fields))
            if self.model == 'mfd':
                valid_field = set(fields).issubset(set(self.mfd_fields))
            if self.model == 'asc':
                valid_field = set(fields).issubset(set(self.asc_fields))
            if not valid_field:
                raise AFLOWmlAPIError(
                    'invalid fields specified, must be from the following \n' +
                    '    plmf: ' + ', '.join(self.plmf_fields) + '\n' +
                    '    mfd: ' + ', '.join(self.mfd_fields) + '\n' +
                    '    asc: ' + ', '.join(self.asc_fields) + '\n'
                )
        else:
            if self.model == 'plmf':
                fields = self.plmf_fields
            if self.model == 'mfd':
                fields = self.mfd_fields
            if self.model == 'asc':
                fields = self.asc_fields

        if self.model is None:
            raise AFLOWmlAPIError(
                'The ML model has not been specified. Please make sure' +
                ' to call the submit_job method before polling.'
            )

        url = self._base_url + '/prediction/result/' + job_id
        req = Request(url)
        res = None

        try:
            res = urlopen(req).read()
        except HTTPError as e:
            raise AFLOWmlAPIError(
                'Failed to poll job: {}'.format(job_id),
                status_code=e.code
            )

        res_json = None
        try:
            res_json = json_loader(res)
        except ValueError:
            raise AFLOWmlAPIError(
                'Unable to parse response, invalid JSON'
            )

        if res_json['status'] == 'SUCCESS':
            self.res_data = {key: res_json[key] for key in fields}
            return self.res_data
        elif res_json['status'] == 'PENDING':
            sleep(3)
            return self.poll_job(job_id, fields=fields)
        elif res_json['status'] == 'STARTED':
            sleep(10)
            return self.poll_job(job_id, fields=fields)
        elif res_json['status'] == 'FAILURE':
            raise AFLOWmlAPIError(
                'The job has failed, please make sure you have a ' +
                'valid POSCAR, composition or job id'
            )
        else:
            raise AFLOWmlAPIError(
                'Failed to poll job: {}'.format(job_id)
            )

    def get_prediction(self, post_data, model, fields=[]):
        '''
        Calls submit_job and poll_job methods to get a prediction.

        Takes the contents of post_data and the model as arguements.

        Returns the prediction results as a dictionary.
        '''
        job_id = self.submit_job(post_data, model)
        return self.poll_job(job_id, fields=fields)
