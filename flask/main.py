from flask import Flask, render_template, request, redirect, url_for, make_response, abort

try:
	# for internal server
	from urlparse import urlparse, urljoin
except:
	# for heroku push:
	from urllib.parse import urlparse, urljoin

import base64
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import seaborn as sns

# from urlparse import urlparse, urljoin

import functools

from math import ceil
import random
import json
import os
import datetime

from forms import DataSelection, FunctionSelection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# import kaggle
# from kaggle.api.kaggle_api_extended import KaggleApi
# api = KaggleApi()
# api.authenticate()

app = Flask(__name__)

app.config['SECRET_KEY'] = os.urandom(24)

def get_info():
	info = {}
	info["sources"] = []
	info["sources"].append({"name":"CDC COVID Dashboard","website":"https://www.cdc.gov/coronavirus/2019-ncov/index.html"})
	info["sources"].append({"name":"CDC U.S. COVID Data Tracker","website":"https://covid.cdc.gov/covid-data-tracker/#cases_totalcases"})
	info["sources"].append({"name":"DataSF COVID-19 Data and Reports","website":"https://data.sfgov.org/stories/s/fjki-2fab"})
	info["sources"].append({"name":"Send a Virtual Hug","website":"http://sendavirtualhug.com"})
	return info


@app.errorhandler(404)
def page_not_found(e):
    title = 'Not Found'
    code = '404'
    message = "We can't seem to find the page you're looking for."
    return render_template('error.html', code = code, message = message, title = title, info = get_info()), 404

@app.errorhandler(403)
def page_forbidden(e):
    title = 'Forbidden'
    code = '403'
    message = "You do not have access to this page."
    return render_template('error.html', code = code, message = message, title = title, info = get_info()), 403

@app.errorhandler(500)
def internal_server_error(e):
    title = 'Internal Server Error'
    code = '500'
    message = "The server encountered an internal error and was unable to complete your request. Either the server is overloaded or there is an error in the application."
    return render_template('error.html', code = code, message = message, title = title, info = get_info()), 500

@app.route('/', methods=['GET', 'POST'])
def index():
	form = DataSelection()
	if form.validate_on_submit():
		print(form.dataset.data)
		dataset = form.dataset.data
		data_url = 'static/data/'+str(dataset)+'.csv'
		return redirect('/select-function?dataset='+dataset)
	else:
		for fieldName, errorMessages in form.errors.items():
			for err in errorMessages:
				print(err)
	if request.method == 'POST':
		print(form.errors.items())
		# and form.validate():
		print(form.data)
		print(form.dataset.data)
		print(request.data)
		print(request.form)
		print(request.files)
		print("*")
		print(request.files['file'])
		print(request.form['file_upload'])
		print(request.files.get('file').filename)
		print(form.file.data.filename)
		filename = form.file.data.filename  
		filestream =  form.file.data 
		filestream.seek(0)
		ef = pd.read_csv(filestream)
		sr = pd.DataFrame(ef)
		print(sr)
		if request.form['dataset']:
			dataset = request.form['dataset']
			data_url = 'static/data/'+str(dataset)+'.csv'
		elif request.form['link']:
			dataset = request.form['link']
		elif request.form['file_upload']:
			print(request.form['file_upload'])
		elif request.files['file']:
			uploaded_file = request.files['file']
			if uploaded_file.filename != '':
				uploaded_file.save(uploaded_file.filename)
				data = pd.read_csv(uploaded_file.filename)
				print(data)
		return redirect('/select-function?dataset='+dataset)
    # form = SearchForm()
	# groups_list=[(i.groupID, i.groupName) for i in available_groups]
	# form.groupID.choices = groups_list
	#return render_template('function.html', info = get_info(), form = form, dataset=)
	return render_template('index.html', info=get_info(), form = form)

@app.route('/select-function', methods=['GET', 'POST'])
def select_function():
	form = FunctionSelection()
	dataset = request.args.get('dataset')
	data = pd.read_csv('static/data/'+str(dataset)+'.csv', thousands=',')
	# parsed = urlparse.urlparse(dataset)
	print(dataset)
	choices = []
	for col in data.columns:
		choices.append((col,col))
		# print(data[col][1].isdigit())
		# print(col)
		# if data[col][1].isdigit():
		# 	data[col] = pd.to_numeric(data[col],errors='coerce')
	form.x.choices = choices
	form.y.choices = choices
	form.t.choices = choices
	form.pie.choices = choices
	# if dataset == "Survey_2020":
	# 	form.x.choices = [('Do you drink coffee most days of the week?', 'Do you drink coffee most days of the week?'), ('Do you prefer standards based grading or percentage grading?', 'Do you prefer standards based grading or percentage grading?'), ('What is your ideal number of minutes for a class?', 'What is your ideal number of minutes for a class?'), ('How many student club would you say you actively participate in?', 'How many student club would you say you actively participate in?'), ('On a scale of 1-10, how is online learning going for you so far this quarter?', 'On a scale of 1-10, how is online learning going for you so far this quarter?'), ('What has been your favorite activity during shelter in place?', 'What has been your favorite activity during shelter in place?'), ('On average, how many hours did you spend outside each weekday during the summer?', 'On average, how many hours did you spend outside each weekday during the summer?'), ('On average, how many hours do you spend outside each week day now that school has started?', 'On average, how many hours do you spend outside each week day now that school has started?'), ('On average, how many hours of sleep did you get per weeknight over the summer?', 'On average, how many hours of sleep did you get per weeknight over the summer?'), ('On average, how many hours of sleep do you get per weeknight now that school has started?', 'On average, how many hours of sleep do you get per weeknight now that school has started?'), ('Would you feel safe at this moment having in person school with 8 students per class?', 'Would you feel safe at this moment having in person school with 8 students per class?'), ('I practice academic integrity (always, sometimes, never)', 'I practice academic integrity (always, sometimes, never)'), ('What is your current grade level?,', 'What is your current grade level?,'), (',Do you identify as POC?', ',Do you identify as POC?'), ('I identify as:', 'I identify as:')]
	# 	form.y.choices = form.x.choices
	# 	form.t.choices = [('On a scale of 1-10, how is online learning going for you so far this quarter?', 'On a scale of 1-10, how is online learning going for you so far this quarter?'), ('On average, how many hours did you spend outside each weekday during the summer?', 'On average, how many hours did you spend outside each weekday during the summer?'), ('On average, how many hours do you spend outside each week day now that school has started?', 'On average, how many hours do you spend outside each week day now that school has started?'), ('On average, how many hours of sleep did you get per weeknight over the summer?', 'On average, how many hours of sleep did you get per weeknight over the summer?'), ('On average, how many hours of sleep do you get per weeknight now that school has started?', 'On average, how many hours of sleep do you get per weeknight now that school has started?')]
	# elif dataset == "Redbook_Survey":
	# 	form.x.choices = [('Do you use your Redbook frequently (3 times a week or more)', 'Do you use your Redbook frequently (3 times a week or more)'), ('How much do you think a Redbook costs per person?', 'How much do you think a Redbook costs per person?'), ('Do you think Redbooks are bad for the environment?', 'Do you think Redbooks are bad for the environment?')]
	# 	form.y.choices = form.x.choices
	if request.method == 'POST':
		# dataset = request.form['dataset']
		if request.form:
			if request.form['function'] == "scatterplot":
				return redirect(url_for('get_results', dataset = dataset, function = request.form['function'],x=request.form['x'],y=request.form['y']))
			elif request.form['function'] == "pie_chart":
				return redirect(url_for('get_results', dataset = dataset, function = request.form['function'],x=request.form['pie']))
			else:
				return redirect(url_for('get_results', dataset = dataset, function = request.form['function'],x=request.form['t'],tail=request.form['t_tail'],mean=request.form['hypothetical_mean']))
			# ax.plot(data[x_input],data[y_input], 'yo', data[x_input], poly1d_fn(data[x_input]), '--k')
			
			print(data)
			
			# return redirect('/results?dataset='+dataset+"&function="+request.form['function'])
		# 	if request.form['function'] == "scatterplot":
		# 		data = data.replace(',','', regex=True)
		# 		x_input = request.form['x']
		# 		y_input = request.form['y']
		# 		fig = Figure()
		# 		ax = fig.subplots()

		# 		if request.form['decimals']:
		# 			num_decimals = int(request.form['decimals'])
		# 		else:
		# 			num_decimals = 3
		# 		# float_arr = np.vstack(data[x_input][:, :]).astype(np.float)
		# 		# print(float_arr)
		# 		# ax.plot([1,2])
		# 		coef = np.polyfit(data[x_input].astype(float).dropna(),data[y_input].astype(float).dropna(),1)
		# 		poly1d_fn = np.poly1d(coef) 
		# 		ax.plot(data[x_input],data[y_input], 'yo', data[x_input], poly1d_fn(data[x_input]), '--k')
		# 		slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_input], data[y_input])
		# 		ax.set_title(x_input+"\n vs. "+y_input, fontsize=8) #title
		# 		ax.set_xlabel(x_input) #x label
		# 		ax.set_ylabel(y_input) #y label
		# 		# Save it to a temporary buffer.
		# 		buf = BytesIO()
		# 		fig.savefig(buf, format="png")
		# 		# Embed the result in the html output.
		# 		data2 = base64.b64encode(buf.getbuffer()).decode("ascii")
		# 		plot = f"<img style='width:100%' src='data:image/png;base64,{data2}'/>"
		# 		fig = Figure()
		# 		ax = fig.subplots()
		# 		sns.residplot(data[x_input],data[y_input],ax=ax)
		# 		ax.set_title("Residual plot: "+x_input+"\n vs. "+y_input, fontsize=8) #title
		# 		ax.set_xlabel("") #x label
		# 		ax.set_ylabel("Residuals") #y label
		# 		buf = BytesIO()
		# 		fig.savefig(buf, format="png")
		# 		data2 = base64.b64encode(buf.getbuffer()).decode("ascii")
		# 		residual_plot = f"<img style='width:100%' src='data:image/png;base64,{data2}'/>"
				
		# 		all_stats = [{"name":"slope","val":round(float(slope),num_decimals)}, {"name":"intercept","val":round(float(intercept),num_decimals)}, {"name":"R","val":round(float(r_value),num_decimals)}, {"name":"R SQUARED","val":str(round(float(r_value**2),num_decimals))+" ("+str(round(float(r_value**2*100),num_decimals))+'% of the variance in "'+x_input+'" can be explained by a linear regression on "'+y_input+'".)'}, {"name":"p","val":p_value}, {"name":"Standard Error","val":round(float(std_err),num_decimals)}]
		# 		# return render_template('index.html', info = get_info(), form = form, residual_plot = residual_plot, all_stats = all_stats)
		# 		code_snippet = """
		# 		coef = np.polyfit(data[x_input].astype(float).dropna(),data[y_input].astype(float).dropna(),1)
		# 		poly1d_fn = np.poly1d(coef) 
		# 		ax.plot(data[x_input],data[y_input], 'yo', data[x_input], poly1d_fn(data[x_input]), '--k')
		# 		slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_input], data[y_input])
		# 		fig = Figure()
		# 		ax = fig.subplots()
		# 		sns.residplot(data[x_input],data[y_input],ax=ax)
		# 		"""
		# 		selection = [{"name":"Home","val":"Select Dataset","link":"/"}, {"name":"Function","val":"Generate Scatterplot & Calculate Summary Statistics","link":"/select-function?dataset="+dataset},{"name":"Results","val":"Results","link":"#"}]
		# 		selected_variables = {"name":"Explanatory Variable","val":x_input},{"name":"Response Variable","val":y_input}
		# 		return render_template('results.html', info = get_info(), form = form, residual_plot = residual_plot, all_stats = all_stats, code_snippet = code_snippet, selection = selection, selected_variables = selected_variables)
		# 	elif request.form['function'] == "1_samp_t_test":
		# 		x_input = request.form['t']
		# 		data_input = pd.to_numeric(data[x_input])
		# 		tscore, pvalue = stats.ttest_1samp(data_input, popmean=float(request.form['hypothetical_mean']))
		# 		if pvalue < 0.05:
		# 			all_stats = [{"name":"t score","val":tscore},{"name":"p-value","val":pvalue},{"name":"p < 0.05","val":"Reject the null hypothesis"}]
		# 		else:
		# 			all_stats = [{"name":"t score","val":tscore},{"name":"p-value","val":pvalue},{"name":"p ≥ 0.05","val":"Do not reject the null hypothesis"}]
		# 		code_snippet = """
		# 		x_input = request.form['t']
		# 		data_input = pd.to_numeric(data[x_input])
		# 		tscore, pvalue = stats.ttest_1samp(data_input, popmean=float(request.form['hypothetical_mean']))
		# 		"""
		# 		selection = [{"name":"Function","val":"1 Sample T-Test"},{"name":"Variable","val":x_input},{"name":"Hypothetical Mean","val":data_input}]
		# 		selection = [{"name":"Home","val":"Select Dataset","link":"/"}, {"name":"Function","val":"1-Sample T-Test","link":"/select-function?dataset="+dataset},{"name":"Results","val":"Results","link":"#"}]
		# 		return render_template('results.html', info = get_info(), form = form, all_stats = all_stats, code_snippet = code_snippet, selection = selection)
		# 	else:
		# 		print('ha')
		# else:
		# 	print('hey')
	return render_template('function.html', info = get_info(), dataset=dataset, form = form)


@app.route('/results', methods=['GET', 'POST'])
def get_results():
	try:
		dataset = request.args.get('dataset')
		function = request.args.get('function')
		if request.args.get('decimals'):
			num_decimals = int(request.args.get('decimals'))
		else:
			num_decimals = 3
	except:
		dataset = "Survey_2020"
		function = "scatterplot"
	# data = pd.read_csv('static/data/'+str(dataset)+'.csv', thousands=',')
	data = pd.read_csv('http://isabellas-independent-study.herokuapp.com/static/data/'+str(dataset)+'.csv', thousands=',')
	if function == "scatterplot":
		data = data.replace(',','', regex=True)
		x_input = request.args.get('x')
		y_input = request.args.get('y')
		fig = Figure()
		ax = fig.subplots()

		
		# float_arr = np.vstack(data[x_input][:, :]).astype(np.float)
		# print(float_arr)
		# ax.plot([1,2])
		coef = np.polyfit(data[x_input].astype(float).dropna(),data[y_input].astype(float).dropna(),1)
		poly1d_fn = np.poly1d(coef) 
		ax.plot(data[x_input],data[y_input], 'yo', data[x_input], poly1d_fn(data[x_input]), '--k')
		slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_input], data[y_input])
		ax.set_title(x_input+"\n vs. "+y_input, fontsize=8) #title
		ax.set_xlabel(x_input) #x label
		ax.set_ylabel(y_input) #y label
		# Save it to a temporary buffer.
		buf = BytesIO()
		fig.savefig(buf, format="png")
		# Embed the result in the html output.
		data2 = base64.b64encode(buf.getbuffer()).decode("ascii")
		plot = f"<img style='width:100%' src='data:image/png;base64,{data2}'/>"
		fig = Figure()
		ax = fig.subplots()
		sns.residplot(data[x_input],data[y_input],ax=ax)
		ax.set_title("Residual plot: "+x_input+"\n vs. "+y_input, fontsize=8) #title
		ax.set_xlabel("") #x label
		ax.set_ylabel("Residuals") #y label
		buf = BytesIO()
		fig.savefig(buf, format="png")
		data2 = base64.b64encode(buf.getbuffer()).decode("ascii")
		residual_plot = f"<img style='width:100%' src='data:image/png;base64,{data2}'/>"
		
		all_stats = [{"name":"slope","val":round(float(slope),num_decimals)}, {"name":"intercept","val":round(float(intercept),num_decimals)}, {"name":"R","val":round(float(r_value),num_decimals)}, {"name":"R SQUARED","val":str(round(float(r_value**2),num_decimals))+" ("+str(round(float(r_value**2*100),num_decimals))+'% of the variance in "'+x_input+'" can be explained by a linear regression on "'+y_input+'".)'}, {"name":"p","val":p_value}, {"name":"Standard Error","val":round(float(std_err),num_decimals)}]
		# return render_template('index.html', info = get_info(), form = form, residual_plot = residual_plot, all_stats = all_stats)
		code_snippet = """
		coef = np.polyfit(data[x_input].astype(float).dropna(),data[y_input].astype(float).dropna(),1)
		poly1d_fn = np.poly1d(coef) 
		ax.plot(data[x_input],data[y_input], 'yo', data[x_input], poly1d_fn(data[x_input]), '--k')
		slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_input], data[y_input])
		fig = Figure()
		ax = fig.subplots()
		sns.residplot(data[x_input],data[y_input],ax=ax)
		"""
		selection = [{"name":"Home","val":"Select Dataset","link":"/"}, {"name":"Function","val":"Generate Scatterplot & Calculate Summary Statistics","link":"/select-function?dataset="+dataset},{"name":"Results","val":"Results","link":"#"}]
		selected_variables = {"name":"Explanatory Variable","val":x_input},{"name":"Response Variable","val":y_input}
		return render_template('results.html', plot=plot, info = get_info(), residual_plot = residual_plot, all_stats = all_stats, code_snippet = code_snippet, selection = selection, selected_variables = selected_variables)
	elif function == "pie_chart":
		x = request.args.get('x')
		series = data[x].value_counts() / len(data)
		labels = [k for item in [series] for k,v in item.items()]
		sizes = [v for item in [series] for k,v in item.items()]
		fig1 = Figure()
		ax1 = fig1.subplots()
		ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
				shadow=True, startangle=90)
		ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
		buf = BytesIO()
		fig1.savefig(buf, format="png")
		data3 = base64.b64encode(buf.getbuffer()).decode("ascii")
		pie_chart = f"<img style='width:100%' src='data:image/png;base64,{data3}'/>"	
		selection = [{"name":"Home","val":"Select Dataset","link":"/"}, {"name":"Function","val":"Generate Scatterplot & Calculate Summary Statistics","link":"/select-function?dataset="+dataset},{"name":"Results","val":"Results","link":"#"}]
		selected_variables = [{"name":"Variable to Test","val":x}]
		code_snippet = """
		series = data[x].value_counts() / len(data)
		labels = [k for item in [series] for k,v in item.items()]
		sizes = [v for item in [series] for k,v in item.items()]
		fig1 = Figure()
		ax1 = fig1.subplots()
		ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
				shadow=True, startangle=90)
		ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
		"""
		return render_template('results.html', pie_chart=pie_chart, info = get_info(), code_snippet = code_snippet, selection = selection, selected_variables = selected_variables)
	elif function == "1_samp_t_test":
		x_input = request.args.get('x')
		mean = request.args.get('mean')
		tail = request.args.get('tail')
		data_input = pd.to_numeric(data[x_input])
		tscore, pvalue = stats.ttest_1samp(data_input, popmean=float(mean))
		if tail=="<" or tail==">":
			pvalue = pvalue/2
		if pvalue < 0.05:
			all_stats = [{"name":"t score","val":tscore},{"name":"p-value","val":pvalue},{"name":"p < 0.05","val":"Reject the null hypothesis"}]
		else:
			all_stats = [{"name":"t score","val":tscore},{"name":"p-value","val":pvalue},{"name":"p ≥ 0.05","val":"Do not reject the null hypothesis"}]
		code_snippet = """
		x_input = request.form['t']
		data_input = pd.to_numeric(data[x_input])
		tscore, pvalue = stats.ttest_1samp(data_input, popmean=float(request.form['hypothetical_mean']))
		"""
		selection = [{"name":"Function","val":"1 Sample T-Test"},{"name":"Variable","val":x_input},{"name":"Hypothetical Mean","val":data_input}]
		selection = [{"name":"Home","val":"Select Dataset: "+dataset,"link":"/"}, {"name":"Function","val":"Select Function: 1-Sample T-Test","link":"/select-function?dataset="+dataset},{"name":"Results","val":"Results","link":"#"}]
		selected_variables = [{"name":"Variable to Test","val":x_input},{"name":"Hypothetical Mean","val":mean},{"name":"Desired Comparison","val":tail}]
		return render_template('results.html', info = get_info(), all_stats = all_stats, code_snippet = code_snippet, selection = selection, selected_variables = selected_variables)
	else:
		print('ha')

@app.route('/internal')
def internal():
	return render_template('index.html', info = get_info(), form = DataSelection())

@app.route('/resources')
def resources():
	info = get_info()
	info["resources"] = [{"link":"https://realpython.com/numpy-tutorial/","title":"NumPy Tutorial: Your First Steps Into Data Science in Python – Real Python"},{"link":"https://repl.it/@IsabellaHochsc1/Independent-Study-NumPy-Tutorial","title":"Forkable Repl.It NumPy Tutorial"},{"link":"https://realpython.com/python-matplotlib-guide/","title":"Python Plotting With Matplotlib (Guide) – Real Python"},{"link":"https://repl.it/@IsabellaHochsc1/Matplotlib-Tutorial","title":"Forkable Repl.It Matplotlib TUtorial"},{"link":"https://web.stanford.edu/class/cs101/","title":"CS101 Introduction to Computing Principles"},{"link":"https://towardsdatascience.com/14-data-science-projects-to-do-during-your-14-day-quarantine-8bd60d1e55e1","title":"14 Data Science Projects to do During Your 14 Day Quarantine"},{"link":"https://repl.it/@IsabellaHochsc1/AustralianWildfireVisualizations","title":"Forkable Repl.It Australian Wildfire Visualizations"},{"link":"https://repl.it/@IsabellaHochsc1/Stat2020Survey","title":"Forkable Repl.It UHS 2020 Survey Data"}]
	resources = []
	for resource in info["resources"]:
		if "repl.it" in resource["link"]:
			resource["iframe"] = '<iframe height="400px" width="100%" src="'+resource["link"]+'?lite=true" scrolling="no" frameborder="no" allowtransparency="true" allowfullscreen="true" sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-modals"></iframe>'
		resources.append(resource)
	info["resources"] = resources
	return render_template('resources.html', info = info)


@app.route('/sitemap.xml', methods=['GET'])
def sitemap():
    """Generate sitemap.xml """
    pages = []
    # All pages registed with flask apps
    for rule in app.url_map.iter_rules():
        if "GET" in rule.methods and len(rule.arguments) == 0:
            pages.append(rule.rule)

    sitemap_xml = render_template('sitemap_template.xml', pages=pages)
    response = make_response(sitemap_xml)
    response.headers["Content-Type"] = "application/xml"

    # return response
    return render_template('sitemap_template.xml', pages=pages)



app.jinja_env.cache = {}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)