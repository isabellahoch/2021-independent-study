try:
	from flask_wtf import Form, FlaskForm
except:
	from flask_wtf import Form
from wtforms import StringField, PasswordField, Form, SelectField, SubmitField, BooleanField, FieldList, FormField, TimeField, RadioField, IntegerField
from wtforms.validators import InputRequired
from wtforms.widgets import TextArea

class DataSelection(Form):
    dataset = SelectField(u'Select Dataset', choices=[('1', 'Select a Dataset'), ('Survey_2020', '2020 UHS Stat Survey (11/12)'), ('Redbook_Survey', 'UHS Redbook Usage Survey')],render_kw={'class': 'form-control form-select','id': 'data-select-field'},validators=[InputRequired()])
    redbook_survey_x = SelectField(u'Select Explanatory Variable', choices=[('Do you use your Redbook frequently (3 times a week or more)', 'Do you use your Redbook frequently (3 times a week or more)'), ('How much do you think a Redbook costs per person?', 'How much do you think a Redbook costs per person?'), ('Do you think Redbooks are bad for the environment?', 'Do you think Redbooks are bad for the environment?')],render_kw={'class': 'form-control form-select redbooksurvey dataset','id': 'redbook-x-select-field'})
    redbook_survey_y = SelectField(u'Select Response Variable', choices=[('Do you use your Redbook frequently (3 times a week or more)', 'Do you use your Redbook frequently (3 times a week or more)'), ('How much do you think a Redbook costs per person?', 'How much do you think a Redbook costs per person?'), ('Do you think Redbooks are bad for the environment?', 'Do you think Redbooks are bad for the environment?')],render_kw={'class': 'form-control form-select redbooksurvey dataset','id': 'redbook-y-select-field'})
    survey_2020_x = SelectField(u'Select Explanatory Variable', choices=[('Do you drink coffee most days of the week?', 'Do you drink coffee most days of the week?'), ('Do you prefer standards based grading or percentage grading?', 'Do you prefer standards based grading or percentage grading?'), ('What is your ideal number of minutes for a class?', 'What is your ideal number of minutes for a class?'), ('How many student club would you say you actively participate in?', 'How many student club would you say you actively participate in?'), ('On a scale of 1-10, how is online learning going for you so far this quarter?', 'On a scale of 1-10, how is online learning going for you so far this quarter?'), ('What has been your favorite activity during shelter in place?', 'What has been your favorite activity during shelter in place?'), ('On average, how many hours did you spend outside each weekday during the summer?', 'On average, how many hours did you spend outside each weekday during the summer?'), ('On average, how many hours do you spend outside each week day now that school has started?', 'On average, how many hours do you spend outside each week day now that school has started?'), ('On average, how many hours of sleep did you get per weeknight over the summer?', 'On average, how many hours of sleep did you get per weeknight over the summer?'), ('On average, how many hours of sleep do you get per weeknight now that school has started?', 'On average, how many hours of sleep do you get per weeknight now that school has started?'), ('ould you feel safe at this moment having in person school with 8 students per class?', 'ould you feel safe at this moment having in person school with 8 students per class?'), ('I practice academic integrity (always, sometimes, never)', 'I practice academic integrity (always, sometimes, never)'), ('What is your current grade level?,', 'What is your current grade level?,'), (',Do you identify as POC?', ',Do you identify as POC?'), ('I identify as:', 'I identify as:')],render_kw={'class': 'form-control form-select 2020survey dataset','id': '2020-survey-x-select-field'})
    survey_2020_y = SelectField(u'Select Response Variable', choices=[('Do you drink coffee most days of the week?', 'Do you drink coffee most days of the week?'), ('Do you prefer standards based grading or percentage grading?', 'Do you prefer standards based grading or percentage grading?'), ('What is your ideal number of minutes for a class?', 'What is your ideal number of minutes for a class?'), ('How many student club would you say you actively participate in?', 'How many student club would you say you actively participate in?'), ('On a scale of 1-10, how is online learning going for you so far this quarter?', 'On a scale of 1-10, how is online learning going for you so far this quarter?'), ('What has been your favorite activity during shelter in place?', 'What has been your favorite activity during shelter in place?'), ('On average, how many hours did you spend outside each weekday during the summer?', 'On average, how many hours did you spend outside each weekday during the summer?'), ('On average, how many hours do you spend outside each week day now that school has started?', 'On average, how many hours do you spend outside each week day now that school has started?'), ('On average, how many hours of sleep did you get per weeknight over the summer?', 'On average, how many hours of sleep did you get per weeknight over the summer?'), ('On average, how many hours of sleep do you get per weeknight now that school has started?', 'On average, how many hours of sleep do you get per weeknight now that school has started?'), ('ould you feel safe at this moment having in person school with 8 students per class?', 'ould you feel safe at this moment having in person school with 8 students per class?'), ('I practice academic integrity (always, sometimes, never)', 'I practice academic integrity (always, sometimes, never)'), ('What is your current grade level?,', 'What is your current grade level?,'), (',Do you identify as POC?', ',Do you identify as POC?'), ('I identify as:', 'I identify as:')],render_kw={'class': 'form-control form-select 2020survey dataset','id': '2020-survey-y-select-field'})
    decimals = IntegerField("Number of Decimals", render_kw = {'placeholder':'enter desired number of decimal places','class':'form-control'})
    submit = SubmitField('Generate Scatterplot & Calculate Summary Statistics', render_kw={'class': 'form-control btn btn-custom','style':'width:100%','style':'margin-top:10px'})
