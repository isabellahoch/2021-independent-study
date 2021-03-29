import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statistical_tests import t_test, t_test_builtin, one_samp_t_test
import seaborn as sns
sns.set(style="white", color_codes=True)

data = pd.read_csv('/static/data/Survey_2020.csv')
print(data.head())

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2

def t_tests_examples():
    print("\n2-SAMPLE T TEST COMPARING AVG SUMMER HOURS OF SLEEP/AVG SCHOOL HOURS OF SLEEP\n")
    print(t_test(data["On average, how many hours of sleep did you get per weeknight over the summer?"],data["On average, how many hours of sleep do you get per weeknight now that school has started?"],len(data["On average, how many hours of sleep did you get per weeknight over the summer?"])))
    print("confirming the method I wrote works using built-in method:")
    t_test_builtin(data["On average, how many hours of sleep did you get per weeknight over the summer?"],data["On average, how many hours of sleep do you get per weeknight now that school has started?"],len(data["On average, how many hours of sleep did you get per weeknight over the summer?"]))


    print("\nCOMPARING AVG PREFERRED CLASS TIME OF SENIORS VS. JUNIORS\n")

    filter_11 = []
    filter_12 = []
    for element in data["What is your current grade level?"]:
        if element == 11:
            filter_11.append(True)
            filter_12.append(False)
        else:
            filter_11.append(False)
            filter_12.append(True)
        
    avg_11 = data["What is your ideal number of minutes for a class?"][filter_11]
    avg_12 = data["What is your ideal number of minutes for a class?"][filter_12]
    print(t_test(avg_12,avg_11,len(avg_11)))

    print("\n1-SAMPLE T TEST: is student online learning progression greater than a 5/10 at 0.05 significance?\n*note: since this is a one-tailed t-test, the true t-value is t/2*\n")

    one_samp_t_test(data["On a scale of 1-10, how is online learning going for you so far this quarter?"],5)


def user_input():
    print("\nRESIDUAL PLOT\n")

    x_input = input("Choose an explanatory variable (x-axis): \nA: Ideal Class Length \nB: Student Club Participation \nC: Online Learning Rating (1-10) \nD: Average Weekday Hours Outside (Summer)\nE: Average Weekday Hours Outside (School)\nF: Average Weeknight Hours of Sleep (Summer)\nG: Average Weeknight Hours of Sleep (School)\n")
    y_input = input("Choose a response variable (y-axis): \nA: Ideal Class Length \nB: Student Club Participation \nC: Online Learning Rating (1-10) \nD: Average Weekday Hours Outside (Summer)\nE: Average Weekday Hours Outside (School)\nF: Average Weeknight Hours of Sleep (Summer)\nG: Average Weeknight Hours of Sleep (School)\n*NOTE: response variable must be different from explanatory variable!*\n")

    x_input = x_input.upper()
    y_input = y_input.upper()

    initial_x = x_input
    initial_y = y_input

    if x_input=="A":
        x_input = "What is your ideal number of minutes for a class?"
    elif x_input=="B":
        x_input = "How many student club would you say you actively participate in?"
    elif x_input=="C":
        x_input = "On a scale of 1-10, how is online learning going for you so far this quarter?"
    elif x_input=="D":
        x_input = "On average, how many hours did you spend outside each weekday during the summer?"
    elif x_input=="E":
        x_input = "On average, how many hours do you spend outside each week day now that school has started?"
    elif x_input=="F":
        x_input = "On average, how many hours of sleep did you get per weeknight over the summer?"
    elif x_input=="G":
        x_input = "On average, how many hours of sleep do you get per weeknight now that school has started?"

    if y_input=="A":
        y_input = "What is your ideal number of minutes for a class?"
    elif y_input=="B":
        y_input = "How many student club would you say you actively participate in?"
    elif y_input=="C":
        y_input = "On a scale of 1-10, how is online learning going for you so far this quarter?"
    elif y_input=="D":
        y_input = "On average, how many hours did you spend outside each weekday during the summer?"
    elif y_input=="E":
        y_input = "On average, how many hours do you spend outside each week day now that school has started?"
    elif y_input=="F":
        y_input = "On average, how many hours of sleep did you get per weeknight over the summer?"
    elif y_input=="G":
        y_input = "On average, how many hours of sleep do you get per weeknight now that school has started?"


def statify(x_input, y_input):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_input], data[y_input])
    print("SLOPE: "+str(slope)+"\tINTERCEPT: "+str(intercept))
    print("R = "+str(r_value)+"\tR^2 = "+str(rsquared(data[x_input], data[y_input])))
    print("p value = "+str(p_value)+"\tSTANDARD ERROR = "+str(std_err))

    # print("R^2 = "+str(rsquared(data[x_input], data[y_input])))
    print(str(rsquared(data[x_input], data[y_input])*100)+"% of the variance in "+initial_x+" can be explained by "+initial_y+".")
    coef = np.polyfit(data[x_input],data[y_input],1)
    poly1d_fn = np.poly1d(coef) 
    # poly1d_fn is now a function which takes in x and returns an estimate for y

    plt.plot(data[x_input],data[y_input], 'yo', data[x_input], poly1d_fn(data[x_input]), '--k')

    sns.residplot(data[x_input],data[y_input])

    # plt.plot(data[x_input], data[y_input], 'o')
    plt.show()