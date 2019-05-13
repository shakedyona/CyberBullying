import shap
import matplotlib.pyplot as pl
import xgboost as xgb


def explain_model(model, X, folder_name):

    # Plot Variable Importance
    xgb.plot_importance(model, importance_type='weight', show_values=True, title='the number of times a feature appears in tree')
    pl.savefig(folder_name + r'\plot_importance_weight.png')
    pl.clf()
    xgb.plot_importance(model, importance_type='gain', show_values=True,
                        title='The average training loss reduction gained when using a feature for splitting')
    pl.savefig(folder_name + r'\plot_importance_gain.png')
    pl.clf()

    # Visualization feature importances with Shap (global)
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    pl.savefig(folder_name + r'\summary_plot_bar.png')
    pl.clf()

    shap.dependence_plot("post_length", shap_values, X, show=False)
    pl.savefig(folder_name + r'\dependence_plot.png')
    pl.clf()

    shap.summary_plot(shap_values, X, show=False)
    pl.savefig(folder_name + r'\summary_plot.png')
    pl.clf()

    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False, link="logit")
    pl.savefig(folder_name + r'\force_plot_0.png')
    pl.clf()
    shap.force_plot(explainer.expected_value, shap_values[-1, :], X.iloc[-1, :], matplotlib=True, show=False,
                    link="logit")
    pl.savefig(folder_name + r'\force_plot_length.png')
    pl.clf()






