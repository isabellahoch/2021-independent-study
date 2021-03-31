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
import seaborn as sns

# from urlparse import urlparse, urljoin

import functools

from math import ceil
import random
import json
import os
import datetime

from forms import DataSelection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
	if request.method == 'POST':
		print(form.errors.items())
		# and form.validate():
		print(form.data)
		print(form.dataset.data)
		print(request.data)
		print(request.form)
		print(request.form['dataset'])
		if request.form and request.form['dataset'] is not None:
			fig = Figure()
			ax = fig.subplots()
			# ax.plot(data[x_input],data[y_input], 'yo', data[x_input], poly1d_fn(data[x_input]), '--k')
			data = pd.read_csv('static/data/'+str(request.form['dataset'])+'.csv')
			print(data)
			if request.form['dataset'] == "Survey_2020":
				x_input = request.form['survey_2020_x']
				y_input = request.form['survey_2020_y']
			elif request.form['dataset'] == "Redbook_Survey":
				x_input = request.form['redbook_survey_x']
				y_input = request.form['redbook_survey_y']
			else:
				x_input = None
				y_input = None
			print(data[x_input])
			num_decimals = int(request.form['decimals'])
			# float_arr = np.vstack(data[x_input][:, :]).astype(np.float)
			# print(float_arr)
			# ax.plot([1,2])
			coef = np.polyfit(data[x_input],data[y_input],1)
			poly1d_fn = np.poly1d(coef) 
			ax.plot(data[x_input],data[y_input], 'yo', data[x_input], poly1d_fn(data[x_input]), '--k')
			slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_input], data[y_input])
			# Save it to a temporary buffer.
			buf = BytesIO()
			fig.savefig(buf, format="png")
			# Embed the result in the html output.
			data2 = base64.b64encode(buf.getbuffer()).decode("ascii")
			plot = f"<img style='width:100%' src='data:image/png;base64,{data2}'/>"

			fig = Figure()
			ax = fig.subplots()
			sns.residplot(data[x_input],data[y_input],ax=ax)
			buf = BytesIO()
			fig.savefig(buf, format="png")
			data2 = base64.b64encode(buf.getbuffer()).decode("ascii")
			residual_plot = f"<img style='width:100%' src='data:image/png;base64,{data2}'/>"
			
			all_stats = [{"name":"slope","val":round(float(slope),num_decimals)}, {"name":"intercept","val":round(float(intercept),num_decimals)}, {"name":"R","val":round(float(r_value),num_decimals)}, {"name":"R SQUARED","val":str(round(float(r_value**2),num_decimals))+" ("+str(round(float(r_value**2*100),num_decimals))+'% of the variance in "'+x_input+'" can be explained by "'+y_input+'".)'}, {"name":"p","val":p_value}, {"name":"Standard Error","val":round(float(std_err),num_decimals)}]
			return render_template('index.html', info = get_info(), form = form, plot = plot, residual_plot = residual_plot, all_stats = all_stats)
    # form = SearchForm()
	return render_template('index.html', info = get_info(), form = form)

@app.route('/internal')
def internal():
	return render_template('index.html', info = get_info(), form = DataSelection())



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