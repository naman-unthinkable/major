from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import time
from ML import process, make_predictions
import os
import shutil
import random, string
import numpy as np

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
   return render_template('index_new.html')

@app.route('/model/<name>', methods=['GET','POST'])
def get_output_file(name):
	print('file-name', name)
	file_name=app.root_path+'/tmp/'+name+'.pkl'
	print('f name', file_name)
	return send_file(file_name, as_attachment=True)

@app.route('/formSubmit2', methods=['POST'])
def fetch_pred():
	if request.method == 'POST':
		print('fetch_pred')
		job_name=request.form['job_name']
		print('job_name', job_name)
		print(request.files.keys())
		pred_file=request.files['to_predict']
		pred_file.save(app.root_path+'/tmp/data_'+job_name)
		model_path=app.root_path+'/tmp/'+job_name+'.pkl'
		data_path=app.root_path+'/tmp/data_'+job_name
		header=None
		try:
			final_pred_array=make_predictions(model_path, data_path, header)
		except ValueError:
			error_msg="Incorrect data file selected. This model wasn't trained on this data format."
			# return ("Incorrect data file selected. This model wasn't trained on this data format.")
			return render_template('error_page.html', msg=error_msg)
		final_path=app.root_path+'/tmp/'+'predictions_'+job_name+'.txt'
		np.savetxt(final_path, final_pred_array)
		return send_file(final_path, as_attachment=True)

@app.route('/formSubmit', methods=['POST'])
def submit():
	if(request.method == 'POST'):

		f = request.files['file']
		job_name=request.form['job_name']+'_'+randomword(5)

		# job_path=app.root_path+'/tmp/'#+job_name
		job_path=app.root_path
		# model_path=job_path+'/tmp/'+job_name+'.pkl'
		model_path=job_path+'/tmp/'+job_name+'.pkl'

		# f.save(app.root_path+'/tmp/data_'+job_name)
		f.save(app.root_path+'/tmp/data_'+job_name)
		try:
			filename, metric_value=process(request.form, job_path, job_name)
		except ValueError:
			error_msg='Target data is non numeric. Regression requires target to be numeric.'
			return render_template('error_page.html', msg=error_msg)
		except KeyError:
			error_msg="The entered column doesn't exist. Please note that the columns are 0 indexed."
			return render_template('error_page.html', msg=error_msg)
		if request.form['train_type']=='reg':
			METRIC_TYPE='RMSE'
		else:
			METRIC_TYPE='Accuracy'
		return render_template('result.html', filename=job_name, metric_value=metric_value,
			metric_type=METRIC_TYPE)

@app.route('/form_Submit1', methods=['POST'])
def train_on_model():
	temp_model=request.files['model']
	temp_data=request.files['to_predict']

	job_name=request.form['job_name']+'_'+randomword(5)

	# model_path=job_path+'/model/'+job_name+'.pkl'
	model_path=app.root_path+'/tmp/'+job_name+'.pkl'
	# data_path=job_path+'/data/'+'data'
	data_path=app.root_path+'/tmp/'+'data_'+job_name

	temp_model.save(model_path)
	temp_data.save(data_path)

	# header=request.form['header_row']
	# if len(header)==0:
	# 	header=None
	# else:
	# 	header=int(header)
	header=None
	try:
		final_pred_array=make_predictions(model_path, data_path, header)
	except ValueError:
		error_msg="This model is not comptable with the test data selected. This model wasn't trained on this data format."
		return render_template('error_page.html', msg=error_msg)
		# return ("This model is not comptable with the test data selected. This model wasn't trained on this data format.")
	final_path=app.root_path+'/tmp/'+'predictions_'+job_name+'.txt' #job_path+'/pred/predictions.txt'
	np.savetxt(final_path, final_pred_array)
	return send_file(final_path, as_attachment=True)

@app.route('/jobFinished', methods=['POST'])
def finish():
	job_name=request.form['job_name']
	# shutil.rmtree(app.root_path+'/jobs/'+job_name)
	return render_template('index_new.html')

if __name__ == '__main__':
   app.run(debug = False, threaded=True)
