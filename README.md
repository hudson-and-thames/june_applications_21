# Skill Set Challenge!

[Hudson & Thames](https://hudsonthames.org/) has provided the following skillset challenge to allow potential researchers to gauge if they have the required skills to take part in the [apprenticeship program](https://hudsonthames.org/apprenticeship-program/).

<div align="center">
  <img src="https://hudsonthames.org/wp-content/uploads/2021/02/LB_Hudson_Thames_ProductLogos_ArbitrageLab-13.png" height="250"><br>
</div>

## Your Mission:
The following assignment is an opportunity for you to highlight your skillset and show us what you are made of! It tests your ability to implement academic research for the broader quantitative finance community, and to do it in style!

### Briefing

<div align="center">
  <img src="https://raw.githubusercontent.com/hudson-and-thames/june_applications_21/master/images/stop_loss_policy.png?token=ABUXNYRXZT6MK6DMS254NXTAUJBYW" height="250"><br>
</div>

Read the following paper: [When do stop-loss rules stop losses?](https://dspace.mit.edu/bitstream/handle/1721.1/114876/Lo_When%20Do%20Stop-Loss.pdf). 

Note: The paper proposes a simple analytical framework to measure the value added or subtracted by stop-loss rules.

In a Jupyter Notebook (python):

1. Download and save your universe of stocks (S&P500 constituents) (Can use Yahoo finance to get shares data. Checkout the [yfinance](https://github.com/ranaroussi/yfinance) package.)(Else you can use [Polygon](https://polygon.io/))
1. Implement Definitions 1, 2, 3 as well as Propositions 1, 2, 3. This would allow you to construct a framework to determine if adding a stop-loss policy would improve the results of the strategy. Your main method should take a set of observations/parameters of the process and return the properties of a stop-loss policy.
1. Create a set of functions/class for the end-user to make use of.
1. Make sure to add docstrings and follow PEP8 code style checks. Have plenty of inline comments, good variable names and don't over complicate things unnecessarily. It should be easy for the user to make use of.
1. Showcase your new Stop-Loss Rules Framework in a Jupyter Notebook and show us some great visualizations (Matplotlib is a bit basic).
1. Generate a series of AR(1) process observations (Equation 14 in the paper) with different rho values and prove empirically that a momentum strategy will benefit from a stop-loss rule, whereas a mean-reverting strategy will not.
1. Add an introduction, body, and conclusion to your Jupyter Notebook showcasing your new implementation. (Use the correct style headers).
1. Bonus points if you add a comparison of stop-loss strategies with various parameters (Fig. 3, 4, 5 in the paper) and draw your conclusions.
1. Make a Pull Request to this repo so that we can evaluate your work. (Create a new folder with your name)
1. Bonus points if you add unit tests (in a separate .py file).
1. Provide a writeup explaining your way-of-work, your design choices, maybe a UML digram, and learnings.
1. Deadline: 23rd of May 2021  

### How Will You be Evaluated?

Being a good researcher is a multivariate problem that can't be reduced to a few variables. You need to be good at: mathematics, statistics, computer science, finance, and communication.

Pay close attention to detail and know that code style, inline comments, and documentation are heavily weighted.

The one consistent theme between all of our top researchers is their ability to go above and beyond what is asked of them. Please take the initiative to highlight your talent by going the extra mile and adding functionality that end users would love!

### Notes
* Your code for the implementation should be contained in a .py file that you import into your notebook. Please don't have large chunks of code in your notebook.
* IDE Choice: PyCharm.
* Save your data with your PR so that we can evaluate it.
* Turn to the previous cohorts' submissions [here](https://github.com/hudson-and-thames/oct_applications) and [here](https://github.com/hudson-and-thames/march_applications_21) for inspiration.

## Institutional - Need to Know

<div align="center">
  <img src="https://hudsonthames.org/wp-content/uploads/2021/01/logo-black-horisontal.png" height="150"><br>
</div>

**Our Mission**

Our mission is to promote the scientific method within investment management by codifying frameworks, algorithms, and best practices to build the world’s first central repository of ready to use implementations and intellectual property.

* **Company Name**: Hudson and Thames Quantitative Research
* **Company Website**: [https://hudsonthames.org/](https://hudsonthames.org/)
* **Locked Achievement**: Researcher
* **Location**: Virtual Team (We are all in different time zones across the world.)
* **Education**: Familiarity with computer science, statistics, and applied maths. We care a lot more about what you can do rather than your exact qualifications.

### Day on Day Activity
* Implement academic research
* Python
* Unit tests
* PEP8
* Continuous integration
* Documentation
* Writing articles
* Public Speaking

### Skills:
* Must speak fluent English. There is a documentation requirement so English is an absolute requirement.
* Python
* Statistics
* Software engineering
* Object Orientated Programming
* Linear Algebra
