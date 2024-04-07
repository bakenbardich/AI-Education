from explainerdashboard import ExplainerDashboard

db = ExplainerDashboard.from_config("abalone_dash.yaml")
db.run(host='0.0.0.0', port=9050, use_waitress=True)
