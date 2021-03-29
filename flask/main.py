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
		fig = Figure()
		ax = fig.subplots()
		# ax.plot(data[x_input],data[y_input], 'yo', data[x_input], poly1d_fn(data[x_input]), '--k')
		data = pd.read_csv('/static/data/'+str(form.dataset.data)+'.csv')
		if form.dataset.data == "Survey_2020":
			x_input = form.survey_2020_x.data
			y_input = form.survey_2020_y.data
		elif form.dataset.data == "Redbook_Survey":
			x_input = form.redbook_survey_x.data
			y_input = form.redbook_survey_y.data
		# ax.plot([1,2])
		ax.plot(data[x_input],data[y_input], 'yo', data[x_input], poly1d_fn(data[x_input]), '--k')
		# Save it to a temporary buffer.
		buf = BytesIO()
		fig.savefig(buf, format="png")
		# Embed the result in the html output.
		data = base64.b64encode(buf.getbuffer()).decode("ascii")
		plot = f"<img src='data:image/png;base64,{data}'/>"
		return render_template('index.html', info = get_info(), form = DataSelection(), plot = plot)
    # form = SearchForm()
	return render_template('index.html', info = get_info(), form = DataSelection())

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