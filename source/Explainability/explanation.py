import shap
import matplotlib.pyplot as pl
from source import Logger


def explain_model(model, X):
    # Visualization for feature importance with SHAP (global)
    logger = Logger.get_logger_instance()
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    logger.save_picture('summary_plot_bar.png')
    shap.dependence_plot("post_length", shap_values[1], X, show=False)
    logger.save_picture('dependence_plot.png')
    shap.summary_plot(shap_values[1], X, show=False)
    logger.save_picture('summary_plot.png')
    shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X.iloc[0, :], matplotlib=True, show=False,
                    link="logit")
    logger.save_picture('force_plot_0.png')


def explain_class(model,  X):
    logger = Logger.get_logger_instance()
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X.iloc[0, :], matplotlib=True, show=False,
                    link="logit")
    logger.save_picture('force_plot_post.png')
