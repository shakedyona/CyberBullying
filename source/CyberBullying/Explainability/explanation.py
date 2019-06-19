import shap
from .. import Logger, utils
import os.path

SOURCE = os.path.abspath(os.path.join(__file__, '../../'))


def explain_model(model, X):
    # Visualization for feature importance with SHAP (global)
    logger = Logger.get_logger_instance()
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    logger.save_picture('summary_plot_bar.png')
    utils.save_picture(os.path.join(SOURCE, 'outputs/summary_plot_bar.png'))
    utils.clear_plot()
    shap.dependence_plot("post_length", shap_values[1], X, show=False)
    logger.save_picture('dependence_plot.png')
    utils.save_picture(os.path.join(SOURCE, 'outputs/dependence_plot.png'))
    utils.clear_plot()
    shap.summary_plot(shap_values[1], X, show=False)
    logger.save_picture('summary_plot.png')
    utils.save_picture(os.path.join(SOURCE, 'outputs/summary_plot.png'))
    utils.clear_plot()


def explain_class(model,  X):
    logger = Logger.get_logger_instance()
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X.iloc[0, :], matplotlib=True, show=False,
                    link="logit")
    logger.save_picture('force_plot_post.png')
    utils.save_picture(os.path.join(SOURCE, 'outputs/force_plot_post.png'))
    utils.clear_plot()
